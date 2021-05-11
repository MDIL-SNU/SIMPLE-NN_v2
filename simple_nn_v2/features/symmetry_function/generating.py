from __future__ import print_function
from __future__ import division
import os, sys
import torch
import numpy as np
from ase import io
from ase import units
import ase
from ._libsymf import lib, ffi
from ...utils import _gen_2Darray_for_ffi, compress_outcar
from ...utils import data_generator


def generate(inputs_full, logfile):
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

    atom_types = inputs_full['atom_types']
    inputs = inputs_full['symmetry_function']
    structure_list = './str_list'
    data_list = './pickle_list'

    data_list_fil = open(data_list, 'w')
    data_idx = 1
    
    # 1. Get structure list from "structure_list" file
    # structures: list of [STRUCTURE_PATH, INDEX_EXP]  ex) [["/PATH/OUTCAR1", "::10"], ["/PATH/OUTCAR2", "::"], ["/PATH/OUTCAR3", "::"]]
    # structure_tag_idx(int list): list of structure tag index of each STRUCTURE_PATH    ex) [1, 2, 2]
    # structure_tags(str list): list of structure tags     ex) ["None", "TAG1", "TAG2"]
    # structure_weights(float list): list of structure weights       
    structures, structure_tag_idx, structure_tags, structure_weights = data_generator.parse_structure_list(logfile, structure_list=structure_list)

    # 2. Get symmetry function parameter dictionary for each atom types
    symf_params_set = _parsing_symf_params(inputs, atom_types)

    # Convert values in ['int'] & ['double'] into C type data, and save as key named of ['int_p'] & ['double_p']
    for element in atom_types:
        symf_params_set[element]['int_p'] = _gen_2Darray_for_ffi(symf_params_set[element]['int'], ffi, 'int')
        symf_params_set[element]['double_p'] = _gen_2Darray_for_ffi(symf_params_set[element]['double'], ffi)

    for item, tag_idx in zip(structures, structure_tag_idx):
        # 3. Load structure information using ase module (format: ase.atoms.Atoms object iterator)
        snapshots = data_generator.load_snapshots(inputs, item, logfile)

        for snapshot in snapshots:
            # 4. Extract structure information from snapshot (atom_num, cart, scale, cell)
            # atom_type_idx(int list): list of type index for each atoms(start from 1)      ex) [1,1,2,2,2,2]
            # type_num(int dic): number of atoms for each types                             ex) {'Si': 2, 'O': 4}
            # type_atom_idx(int list dic): list of atoms index that for each atom types     ex) {'Si': [0,1], 'O': [2,3,4,5]}
            atom_num, atom_type_idx, type_num, type_atom_idx, cart, scale, cell = _get_structure_info(snapshot, atom_types)

            # Convert atom_type_idx, cart, scale, cell into C type data
            atom_type_idx_p = ffi.cast('int *', atom_type_idx.ctypes.data)
            cart_p  = _gen_2Darray_for_ffi(cart, ffi)
            scale_p = _gen_2Darray_for_ffi(scale, ffi)
            cell_p  = _gen_2Darray_for_ffi(cell, ffi)

            # Initialize result dictionary
            result = _init_result(type_num, structure_tags, structure_weights, tag_idx, atom_type_idx)
            
            for _, jtem in enumerate(atom_types):    
                # Initialize variables for calculation
                # cal_atom_idx(int list): atom index for calculation    ex) [2,3,4]
                # cal_atom_num(int): atom numbers for calculation       ex) 3
                cal_atom_idx, cal_atom_num, x, dx, da = _init_sf_variables(type_atom_idx,\
                    jtem, symf_params_set, atom_num)
                
                # Convert cal_atom_idx, x, dx, da into C type data
                cal_atom_idx_p = ffi.cast('int *', cal_atom_idx.ctypes.data)
                x_p = _gen_2Darray_for_ffi(x, ffi)
                dx_p = _gen_2Darray_for_ffi(dx, ffi)
                da_p = _gen_2Darray_for_ffi(da, ffi)        

                # 5. Calculate symmetry functon using C type data
                errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                                    atom_type_idx_p, atom_num, cal_atom_idx_p, cal_atom_num, \
                                    symf_params_set[jtem]['int_p'], symf_params_set[jtem]['double_p'], symf_params_set[jtem]['num'], \
                                    x_p, dx_p, da_p)

                # Check error occurs
                #self._check_error(errno, logfile)

                # Set result to dictionary format from calculated value
                _set_result(result, x, dx, da, type_num, jtem, symf_params_set, atom_num)
            # End of for loop
            
            # Extract E, F, S from snapshot and append to result dictionary
            E, F, S = _extract_EFS(inputs_full, inputs, snapshot, logfile)
            # Set result from _extract_EFS
            result['E'] = torch.tensor(E)
            result['F'] = torch.tensor(F)
            result['S'] = torch.tensor(S)

            # 6. Save "result" data to pickle file
            tmp_filename = data_generator.save_to_datafile(inputs, result, data_idx, logfile)
            data_list_fil.write("{}:{}\n".format(tag_idx, tmp_filename))
            data_idx += 1
            tmp_endfile = tmp_filename

        logfile.write(': ~{}\n'.format(tmp_endfile))

    data_list_fil.close()

# Get symmetry function parameter list for each atom types
def _parsing_symf_params(inputs, atom_types):
    symf_params_set = dict()
    for element in atom_types:
        symf_params_set[element] = dict()
        symf_params_set[element]['int'], symf_params_set[element]['double'] = \
            __read_params(inputs['params'][element])
        symf_params_set[element]['total'] = np.concatenate((symf_params_set[element]['int'], symf_params_set[element]['double']), axis=1)
        symf_params_set[element]['num'] = len(symf_params_set[element]['total'])            
    return symf_params_set

def __read_params(filename):
    params_int = list()
    params_double = list()
    with open(filename, 'r') as fil:
        for line in fil:
            tmp = line.split()
            params_int += [list(map(int, tmp[:3]))]
            params_double += [list(map(float, tmp[3:]))]

    params_int = np.asarray(params_int, dtype=np.intc, order='C')
    params_double = np.asarray(params_double, dtype=np.float64, order='C')

    return params_int, params_double

# Extract structure information from snapshot (atom numbers, cart, scale, cell)
# Return variables related to structure information (atom_type_idx, type_num, type_atom_idx)
def _get_structure_info(snapshot, atom_types):
    cart = np.copy(snapshot.get_positions(wrap=True), order='C')
    scale = np.copy(snapshot.get_scaled_positions(), order='C')
    cell = np.copy(snapshot.cell, order='C')

    symbols = np.array(snapshot.get_chemical_symbols())
    atom_num = len(symbols)
    atom_type_idx = np.zeros([len(symbols)], dtype=np.intc, order='C')
    type_num = dict()
    type_atom_idx = dict()
    for j, jtem in enumerate(atom_types):
        tmp = symbols==jtem
        atom_type_idx[tmp] = j+1
        type_num[jtem] = np.sum(tmp).astype(np.int64)
        # if atom indexs are sorted by atom type,
        # indexs are sorted in this part.
        # if not, it could generate bug in training process for force training
        type_atom_idx[jtem] = np.arange(atom_num)[tmp]

    return  atom_num, atom_type_idx, type_num, type_atom_idx, cart, scale, cell

# Init result Dictionary 
def _init_result(type_num, structure_tags, structure_weights, idx, atom_type_idx):
    result = dict()
    result['x'] = dict()
    result['dx'] = dict()
    result['da'] = dict()
    result['dx_size'] = dict() ## ADDED
    result['total'] = None ## ADDED
    result['num'] = None ## ADDED
    result['params'] = dict()
    result['N'] = type_num
    result['tot_num'] = np.sum(list(type_num.values()))
    result['struct_type'] = structure_tags[idx]
    result['struct_weight'] = structure_weights[idx]
    result['atom_idx'] = atom_type_idx
    return result
    
# Get data to make C array from variables
def _init_sf_variables(type_atom_idx, jtem, symf_params_set, atom_num, mpi_range=None ):
    if mpi_range != None: # MPI calculation
        cal_atom_idx = np.asarray(type_atom_idx[jtem][mpi_range[0]:mpi_range[1]], dtype=np.intc, order='C')
    elif mpi_range == None: # Serial calculation
        cal_atom_idx = np.asarray(type_atom_idx[jtem], dtype=np.intc, order='C')
    cal_atom_num = len(cal_atom_idx)

    x = np.zeros([cal_atom_num, symf_params_set[jtem]['num']], dtype=np.float64, order='C')
    dx = np.zeros([cal_atom_num, symf_params_set[jtem]['num'] * atom_num * 3], dtype=np.float64, order='C')
    da = np.zeros([cal_atom_num, symf_params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C')

    da = np.zeros([cal_atom_num, symf_params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C')

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
def _set_result(result, x, dx, da, type_num, jtem, symf_params_set, atom_num):
    if type_num[jtem] != 0:
        result['x'][jtem] = np.array(x)
        result['dx'][jtem] = np.array(dx)
        result['da'][jtem] = np.array(da)
        result['x'][jtem] = np.concatenate(result['x'][jtem], axis=0).\
                            reshape([type_num[jtem], symf_params_set[jtem]['num']])
        result['dx'][jtem] = np.concatenate(result['dx'][jtem], axis=0).\
                            reshape([type_num[jtem], symf_params_set[jtem]['num'], atom_num, 3])
        result['da'][jtem] = np.concatenate(result['da'][jtem], axis=0).\
                            reshape([type_num[jtem], symf_params_set[jtem]['num'], 3, 6])
    else:
        result['x'][jtem] = np.zeros([0, symf_params_set[jtem]['num']])
        result['dx'][jtem] = np.zeros([0, symf_params_set[jtem]['num'], atom_num, 3])
        result['da'][jtem] = np.zeros([0, symf_params_set[jtem]['num'], 3, 6])
        #For sparse tensor torch.tensor mappling need
    result['x'][jtem] = torch.tensor(result['x'][jtem])
    result['dx'][jtem] = torch.tensor(result['dx'][jtem])
    result['da'][jtem] = torch.tensor(result['da'][jtem])
    result['params'][jtem] = symf_params_set[jtem]['total']

# Check ase version, E, F, S extract from snapshot, Raise Error 
def _extract_EFS(inputs_full, inputs, snapshot, logfile):
    if not (inputs['refdata_format']=='vasp' or inputs['refdata_format']=='vasp-xdatcar'):
        if ase.__version__ >= '3.18.0':
            E = snapshot.get_potential_energy(force_consistent=True)
        else:
            E = snapshot.get_total_energy()
        try:
            F = snapshot.get_forces()
        except:
            if inputs_full['neural_network']['use_force']:
                err = "There is not force information! Set 'use_force' = false"
                logfile.write("\nError: {:}\n".format(err))
                raise NotImplementedError(err)
        try:
            S = -snapshot.get_stress()/units.GPa*10
            # ASE returns the stress tensor by voigt order xx yy zz yz zx xy
            S = S[[0, 1, 2, 5, 3, 4]]
        except:
            if inputs_full['neural_network']['use_stress']:
                err = "There is not stress information! Set 'use_stress' = false"
                logfile.write("\nError: {:}\n".format(err))
                raise NotImplementedError(err)
    return E, F, S


