from __future__ import print_function
from __future__ import division
import os, sys
import torch
import numpy as np
from ase import io
from ase import units
import ase
from ._libsymf import lib, ffi
from simple_nn_v2.utils import data_generator
from simple_nn_v2.utils import cffi as utils_ffi

from simple_nn_v2.features.symmetry_function import utils  as utils_symf

from simple_nn_v2.features.symmetry_function.mpi import DummyMPI, MPI4PY


def generate(inputs, logfile):
    """ Generate structure data files(format: pickle/pt) that listed in "structure_list" file

    1. Get structure list from parsing "structure_list" file
    2. Get symmetry function parameter list for each atom types from parsing "params_XX" file
        symf_params_set[element] keys:
            'num': total number of symmetry functions
            'total': full parameters in file
            'int': int value parameters in file
            'double': double value parameters in file
            'int_p': convert int value parameters to C type array
            'double_p': convert dobule value parameters to C type array
    3. Load structure information using ase module (format: ase.atoms.Atoms object iterator)
    4. Extract structure information from snapshot
    5. Calculate symmetry functon values using C implemented code
    6. Save to data file (format: pickle or pt)
    """

    atom_types = inputs['atom_types']
    structure_list = './structure_list'
    data_list = './total_list'
    
    #Load MPI 
    try: 
        comm = MPI4PY()
        assert comm.size != 1
        logfile.write("Use mpi in generate size {0}\n".format(comm.size))
    except:
        comm = DummyMPI()
        logfile.write("Not use mpi in generate\n")

    if comm.rank == 0:
        data_list_fil = open(data_list, 'w')
    data_idx = 1
    
    # structure_tag_idx(int list): list of structure tag index of each structure file    ex) [1, 2, 2]  
    structure_tags, structure_weights, structure_file_list, structure_slicing_list, structure_tag_idx  = \
                                data_generator.parse_structure_list(logfile, structure_list=structure_list, comm=comm)

    symf_params_set = utils_symf._parse_symmetry_function_parameters(inputs, atom_types)

    # Convert values into C type data
    for element in atom_types:
        symf_params_set[element]['int_p'] = utils_ffi._gen_2Darray_for_ffi(symf_params_set[element]['int'], ffi, 'int')
        symf_params_set[element]['double_p'] = utils_ffi._gen_2Darray_for_ffi(symf_params_set[element]['double'], ffi)

    for structure_file, structure_slicing, tag_idx in zip(structure_file_list, structure_slicing_list, structure_tag_idx):
        structures = data_generator.load_structures(inputs, structure_file, structure_slicing, logfile, comm)

        for structure in structures:
            # atom_type_idx(int list): list of type index for each atoms(start from 1)      ex) [1,1,2,2,2,2]
            # type_num(int dic): number of atoms for each types                             ex) {'Si': 2, 'O': 4}
            # type_atom_idx(int list dic): list of atoms index for each atom types     ex) {'Si': [0,1], 'O': [2,3,4,5]}
            cell, cart, scale = _get_structure_coordination_info(structure)
            atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type = _get_atom_types_info(structure, atom_types)

            # Convert values into C type data
            atom_type_idx_p = ffi.cast('int *', atom_type_idx.ctypes.data)
            cart_p  = utils_ffi._gen_2Darray_for_ffi(cart, ffi)
            scale_p = utils_ffi._gen_2Darray_for_ffi(scale, ffi)
            cell_p  = utils_ffi._gen_2Darray_for_ffi(cell, ffi)

            result = _initialize_result(atoms_per_type, structure_tags, structure_weights, tag_idx, atom_type_idx)
            
            for idx, element in enumerate(atom_types):

                #MPI part
                mpi_quotient  = atoms_per_type[element] // comm.size
                mpi_remainder = atoms_per_type[element] %  comm.size
                #MPI index     
                begin_idx = comm.rank * mpi_quotient + min(comm.rank , mpi_remainder) 
                end_idx = begin_idx + mpi_quotient 
                #Distribute mpi_remainder to mpi 
                if mpi_remainder > comm.rank:
                    end_idx += 1


                # cal_atom_idx(int list): atom index for calculation    ex) [2,3,4]
                # cal_atom_num(int): atom numbers for calculation       ex) 3
                cal_atom_idx, cal_atom_num, x, dx, da = _initialize_symmetry_function_variables(atom_idx_per_type,\
                    element, symf_params_set, atom_num, mpi_range = (begin_idx,end_idx) )
                
                # Convert cal_atom_idx, x, dx, da into C type data
                cal_atom_idx_p = ffi.cast('int *', cal_atom_idx.ctypes.data)
                x_p = utils_ffi._gen_2Darray_for_ffi(x, ffi)
                dx_p = utils_ffi._gen_2Darray_for_ffi(dx, ffi)
                da_p = utils_ffi._gen_2Darray_for_ffi(da, ffi)        

                # 5. Calculate symmetry functon using C type data
                errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                                    atom_type_idx_p, atom_num, cal_atom_idx_p, cal_atom_num, \
                                    symf_params_set[element]['int_p'], symf_params_set[element]['double_p'], symf_params_set[element]['num'], \
                                    x_p, dx_p, da_p)
                comm.barrier()
                errnos = comm.gather(errno) #List of error number
                errnos = comm.bcast(errnos)
                for errno in errnos:
                    if comm.rank == 0:
                        if errno == 1:
                            err = "Not implemented symmetry function type."
                            logfie.write("\nError: {:}\n".format(err))
                            raise NotImplementedError(err)
                        elif errno == 2:
                            err = "Zeta in G4/G5 must be greater or equal to 1.0."
                            logfie.write("\nError: {:}\n".format(err))
                            raise ValueError(err)
                        else:
                            assert errno == 0, "Unexpected error occred"             
        

                _set_calculated_result(inputs, result, x, dx, da, atoms_per_type, element, symf_params_set, atom_num, comm)
            
            E, F, S = _extract_EFS(inputs, structure, logfile, comm)
            result['E'] = torch.tensor(E)
            if inputs['neural_network']['use_force'] is True:
                result['F'] = torch.tensor(F)
            if inputs['neural_network']['use_stress'] is True:
                result['S'] = torch.tensor(S)

            if comm.rank == 0:
                tmp_filename = data_generator.save_to_datafile(inputs, result, data_idx, logfile)
                data_list_fil.write("{}:{}\n".format(tag_idx, tmp_filename))
                data_idx += 1
                tmp_endfile = tmp_filename
        
        if comm.rank == 0: 
            logfile.write(': ~{}\n'.format(tmp_endfile))
    
    if comm.rank == 0:
        data_list_fil.close()
        if inputs['descriptor']['compress_outcar']:
            os.remove('./tmp_comp_OUTCAR')


# Extract structure information from structure (atom numbers, cart, scale, cell)
# Return variables related to structure information (atom_type_idx, type_num, type_atom_idx)
def _get_structure_coordination_info(structure):
    cell = np.copy(structure.cell, order='C')
    cart = np.copy(structure.get_positions(wrap=True), order='C')
    scale = np.copy(structure.get_scaled_positions(), order='C')

    return cell, cart, scale

def _get_atom_types_info(structure, atom_types):
    # if atom indexs are sorted by atom type,
    # indexs are sorted in this part.
    # if not, it could generate bug in training process for force training
    symbols = np.array(structure.get_chemical_symbols())
    atom_type_idx = np.zeros([len(symbols)], dtype=np.intc, order='C')
    atoms_per_type = dict()
    atom_idx_per_type = dict()

    atom_num = len(symbols)
    for j, element in enumerate(atom_types):
        tmp = symbols==element
        atom_type_idx[tmp] = j+1
        atoms_per_type[element] = np.sum(tmp).astype(np.int64)
        atom_idx_per_type[element] = np.arange(atom_num)[tmp]

    return atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type

# Init result Dictionary 
def _initialize_result(type_num, structure_tags, structure_weights, idx, atom_type_idx):
    result = dict()
    result['x'] = dict()
    result['dx'] = dict()
    result['da'] = dict()
    result['dx_size'] = dict() # due to sparse tensor
    result['total'] = None 
    result['num'] = None 
    result['N'] = type_num
    result['tot_num'] = np.sum(list(type_num.values()))
    result['struct_type'] = structure_tags[idx]
    result['struct_weight'] = structure_weights[idx]
    result['atom_idx'] = atom_type_idx
    return result
    
# Get data to make C array from variables
def _initialize_symmetry_function_variables(type_atom_idx, element, symf_params_set, atom_num, mpi_range=None ):
    if mpi_range != None: # MPI calculation
        cal_atom_idx = np.asarray(type_atom_idx[element][mpi_range[0]:mpi_range[1]], dtype=np.intc, order='C')
    elif mpi_range == None: # Serial calculation
        cal_atom_idx = np.asarray(type_atom_idx[element], dtype=np.intc, order='C')
    cal_atom_num = len(cal_atom_idx)

    x = np.zeros([cal_atom_num, symf_params_set[element]['num']], dtype=np.float64, order='C')
    dx = np.zeros([cal_atom_num, symf_params_set[element]['num'] * atom_num * 3], dtype=np.float64, order='C')
    da = np.zeros([cal_atom_num, symf_params_set[element]['num'] * 3 * 6], dtype=np.float64, order='C')

    da = np.zeros([cal_atom_num, symf_params_set[element]['num'] * 3 * 6], dtype=np.float64, order='C')

    return cal_atom_idx, cal_atom_num, x, dx, da

def _check_error(errnos, logfile):   
    for errno in errnos:
        if errno == 1:
            err = "Not implemented symmetry function type."
            logfile.write("\nError: {:}\n".format(err))
            raise NotImplementedError(err)
        elif errno == 2:
            err = "Zeta in G4/G5 must be greater or equal to 1.0."
            logfile.write("\nError: {:}\n".format(err))
            raise ValueError(err)
        else:
            assert errno == 0    

# Set resulatant Dictionary
def _set_calculated_result(inputs, result, x, dx, da, type_num, element, symf_params_set, atom_num, comm):
    if type_num[element] != 0:
        result['x'][element] = np.array(comm.gather(x,root=0))
        result['dx'][element] = np.array(comm.gather(dx,root=0))
        result['da'][element] = np.array(comm.gather(da,root=0))
        if comm.rank == 0:
            result['x'][element] = np.concatenate(result['x'][element], axis=0).\
                            reshape([type_num[element], symf_params_set[element]['num']])
            result['dx'][element] = np.concatenate(result['dx'][element], axis=0).\
                            reshape([type_num[element], symf_params_set[element]['num'], atom_num, 3])
            result['da'][element] = np.concatenate(result['da'][element], axis=0).\
                            reshape([type_num[element], symf_params_set[element]['num'], 3, 6])
    else:
        result['x'][element] = np.zeros([0, symf_params_set[element]['num']])
        result['dx'][element] = np.zeros([0, symf_params_set[element]['num'], atom_num, 3])
        result['da'][element] = np.zeros([0, symf_params_set[element]['num'], 3, 6])
    if  comm.rank == 0:
        result['x'][element] = torch.tensor(result['x'][element])
        #Sparse tensor mapping here
        if inputs['descriptor']['dx_save_sparse']:
            tmp_tensor = torch.tensor(result['dx'][element])
            result['dx_size'][element] = tmp_tensor.size()
            result['dx'][element] = tmp_tensor.reshape(-1).to_sparse()
            tmp_tensor = None
        else:
            result['dx'][element] = torch.tensor(result['dx'][element])
        result['da'][element] = torch.tensor(result['da'][element])

# Check ase version, E, F, S extract from structure, Raise Error 
def _extract_EFS(inputs, structure, logfile, comm):
    E = None
    F = None
    S = None

    if inputs['descriptor']['refdata_format']=='vasp-out':
        if ase.__version__ >= '3.18.0':
            E = structure.get_potential_energy(force_consistent=True)
        else:
            E = structure.get_total_energy()

        if inputs['neural_network']['use_force'] is True:
            try:
                F = structure.get_forces()
            except:
                err = "There is not force information! Set 'use_force' = false"
                if comm.rank == 0:
                    logfile.write("\nError: {:}\n".format(err))
                raise NotImplementedError(err)

        if inputs['neural_network']['use_stress'] is True:
            try:
                # ASE returns the stress tensor by voigt order xx yy zz yz zx xy
                S = -structure.get_stress()/units.GPa*10
                S = S[[0, 1, 2, 5, 3, 4]]
            except:
                err = "There is not stress information! Set 'use_stress' = false"
                if comm.rank == 0:
                    logfile.write("\nError: {:}\n".format(err))
                raise NotImplementedError(err)

    return E, F, S


