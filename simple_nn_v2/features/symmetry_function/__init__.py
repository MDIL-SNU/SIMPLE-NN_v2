from __future__ import print_function
from __future__ import division
import os, sys
import tensorflow as tf
import numpy as np
import six
from six.moves import cPickle as pickle
from ase import io
from ase import units
import ase
from ._libsymf import lib, ffi
from ...utils import _gen_2Darray_for_ffi, compress_outcar, _generate_scale_file, \
                     _make_full_featurelist, _make_data_list, _make_str_data_list, pickle_load
from ...utils import graph as grp
#from ...utils.mpiclass import DummyMPI, MPI4PY
from braceexpand import braceexpand
from sklearn.decomposition import PCA
from ...utils.datagenerator import Data_generator


class Symmetry_function(object):    
    def __init__(self, inputs=None):
        self.parent = None
        self.key = 'symmetry_function'   # no used
        self.default_inputs = {'symmetry_function': 
                                  {
                                      'params': dict(),
                                      'refdata_format':'vasp-out',
                                      'compress_outcar':True,
                                      'data_per_tfrecord': 150,
                                      'valid_rate': 0.1,
                                      'shuffle':True,
                                      'add_NNP_ref': False, # atom E to tfrecord
                                      'remain_pickle': False,
                                      'continue': False,
                                      'add_atom_idx': True, # For backward compatability
                                      'num_parallel_calls': 5,
                                      'atomic_weights': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'weight_modifier': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'scale_type': 'minmax',
                                      'scale_scale': 1.0,
                                      'scale_rho': None,
                                      'save_to_pickle':False,
                                      'save_directory':'./data'
                                  }
                              }
        self.structure_list = './str_list'
        self.pickle_list = './pickle_list'
        self.train_data_list = './train_list'
        self.valid_data_list = './valid_list'
        self.comm = None
    
    def set_inputs(self):
        self.inputs = self.parent.inputs['symmetry_function']

    # Genreate Method 
    def generate(self):
        """ Generate structure data files(format: pickle/pt) that listed in "structure_list" file

        1. Get structure list from "structure_list" file from parsing "structure_list" file
        2. Get symmetry function parameter list for each atom types from parsing "params_XX" file
            symf_params_set[element] keys:
                'num': total number of symmetry functions
                'total': full parameters in file
                'i': int value parameters in file
                'd': double value parameters in file
                'ip': convert int value parameters to c type array
                'dp': convert dobule value parameters to c type array
        3. Load structure information using ase module (format: ase.atoms.Atoms object iterator)
        4. Extract structure information from snapshot
        5. Calculate symmetry functon values using C implemented code
            result keys:
                'x': 
                'dx': 
                'da': 
                'params': 
                'N': 
                'tot_num': 
                'partition': 
                'partition_XX': 
                'struct_type': 
                'struct_weight': 
                'atom_idx': 
        6. Save to data file (format: pickle or pt)

        """
        
        # Data_generator object for handling [str_list], OUTCAR files, pickle/pt files
        #data_generator = Data_generator(self.inputs, self.structure_list, self.pickle_list, self.parent)
        data_generator = Data_generator(self.inputs, self.parent.logfile, self.structure_list, self.pickle_list)
        
        # 1. Get structure list from "structure_list" file
        # structures: list of [STRUCTURE_PATH, INDEX_EXP]  ex) [["/PATH/OUTCAR1", "::10"], ["/PATH/OUTCAR2", "::"], ["/PATH/OUTCAR3", "::"]]
        # structure_tag_idx(int list): list of structure tag index of each STRUCTURE_PATH    ex) [1, 2, 2]
        # structure_tags(str list): list of structure tags     ex) ["None", "TAG1", "TAG2"]
        # structure_weights(float list): list of structure weights       
        structures, structure_tag_idx, structure_tags, structure_weights = data_generator.parse_structure_list()

        # 2. Get symmetry function parameter list for each atom types
        symf_params_set = self._parsing_symf_params()

        # Parsing C type symmetry function parameter to symf_params_set dictionary
        for element in self.parent.inputs['atom_types']:
            symf_params_set[element]['ip'] = _gen_2Darray_for_ffi(symf_params_set[element]['i'], ffi, "int")
            symf_params_set[element]['dp'] = _gen_2Darray_for_ffi(symf_params_set[element]['d'], ffi)

        for item, tag_idx in zip(structures, structure_tag_idx):
            # 3. Load structure information using ase module (format: ase.atoms.Atoms object iterator)
            snapshots = data_generator.load_snapshots(item)

            for snapshot in snapshots:
                # 4. Extract structure information from snapshot (atom_num, cart, scale, cell)
                atom_num, atom_type_idx, type_num, type_idx, cart, scale, cell = self._get_structure_info(snapshot)

                # Make C type data & 2D array from atom_type_idx, cart, scale, cell
                atom_type_idx_p = ffi.cast("int *", atom_type_idx.ctypes.data)
                cart_p  = _gen_2Darray_for_ffi(cart, ffi)
                scale_p = _gen_2Darray_for_ffi(scale, ffi)
                cell_p  = _gen_2Darray_for_ffi(cell, ffi)

                # Initialize result dictionary
                result = self._init_result(type_num, structure_tags, structure_weights, tag_idx, atom_type_idx)
                
                for _ ,jtem in enumerate(self.parent.inputs['atom_types']):    
                    # Set number of MPI 
                    #begin , end = self._set_mpi(type_num , jtem)
                    #cal_atom_num , cal_atom_idx_p , x , dx , da , x_p , dx_p , da_p = self._get_sf_input(type_idx ,\
                    # jtem  , symf_params_set , atom_num , [begin , end] )

                    # Initialize variables for calculation
                    # cal_atom_idx(int list): atom index for calculation    ex) [2,3,4]
                    # cal_atom_num(int): atom numbers for calculation       ex) 3
                    cal_atom_idx, cal_atom_num,  x, dx, da = self._init_sf_variables(type_idx,\
                     jtem, symf_params_set, atom_num)
                   
                    # Make C array from x, dx, da
                    cal_atom_idx_p = ffi.cast("int *", cal_atom_idx.ctypes.data)
                    x_p = _gen_2Darray_for_ffi(x, ffi)
                    dx_p = _gen_2Darray_for_ffi(dx, ffi)
                    da_p = _gen_2Darray_for_ffi(da, ffi)        

                    # 5. Calculate symmetry functon using C type data
                    errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                                     atom_type_idx_p, atom_num, cal_atom_idx_p, cal_atom_num, \
                                     symf_params_set[jtem]['ip'], symf_params_set[jtem]['dp'], symf_params_set[jtem]['num'], \
                                     x_p, dx_p, da_p)
                    #comm.barrier()
                    #errnos = comm.gather(errno)
                    #errnos = comm.bcast(errnos)

                    # Check error occurs
                    #self._check_error(errno)

                    # Set result to dictionary format from calculated value
                    self._set_result(result, x , dx, da, type_num, jtem, symf_params_set, atom_num)
                # End of for loop
                
                # Extract E, F, S from snapshot and append to result dictionary
                E, F, S = self._extract_EFS(result, snapshot)
                # Set result from _extract_EFS
                result['E'] = E
                result['F'] = F
                result['S'] = S

                # 6. Save "result" data to pickle file
                # ...Need append option for continue generate...
                # ...Need append option for select directory...
                #tmp_endfile = data_generator.save_to_pickle(result, tag_idx, save_dir='./data')
                tmp_endfile = data_generator.save_to_pickle(result, tag_idx)

            self.parent.logfile.write(': ~{}\n'.format(tmp_endfile))

    # Get symmetry function parameter list for each atom types
    def _parsing_symf_params(self):
        symf_params_set = dict()
        for element in self.parent.inputs['atom_types']:
            symf_params_set[element] = dict()
            symf_params_set[element]['i'], symf_params_set[element]['d'] = \
                self.__read_params(self.inputs['params'][element])
            symf_params_set[element]['total'] = np.concatenate((symf_params_set[element]['i'], symf_params_set[element]['d']), axis=1)
            symf_params_set[element]['num'] = len(symf_params_set[element]['total'])            
        return symf_params_set
    
    def __read_params(self, filename):
        params_i = list()
        params_d = list()
        with open(filename, 'r') as fil:
            for line in fil:
                tmp = line.split()
                params_i += [list(map(int, tmp[:3]))]
                params_d += [list(map(float, tmp[3:]))]

        params_i = np.asarray(params_i, dtype=np.intc, order='C')
        params_d = np.asarray(params_d, dtype=np.float64, order='C')

        return params_i, params_d

    # Extract structure information from snapshot (atom numbers, cart, scale, cell)
    # Return variables related to structure information (atom_type_idx, type_num, type_idx)
    def _get_structure_info(self, snapshot):
        cart = np.copy(snapshot.get_positions(wrap=True), order='C')
        scale = np.copy(snapshot.get_scaled_positions(), order='C')
        cell = np.copy(snapshot.cell, order='C')

        symbols = np.array(snapshot.get_chemical_symbols())
        atom_num = len(symbols)
        atom_type_idx = np.zeros([len(symbols)], dtype=np.intc, order='C')
        type_num = dict()
        type_idx = dict()
        for j,jtem in enumerate(self.parent.inputs['atom_types']):
            tmp = symbols==jtem
            atom_type_idx[tmp] = j+1
            type_num[jtem] = np.sum(tmp).astype(np.int64)
            # if atom indexs are sorted by atom type,
            # indexs are sorted in this part.
            # if not, it could generate bug in training process for force training
            type_idx[jtem] = np.arange(atom_num)[tmp]

        return  atom_num, atom_type_idx, type_num, type_idx , cart , scale , cell

    # Init result Dictionary 
    def _init_result(self, type_num, structure_tags, structure_weights, idx, atom_type_idx):
        result = dict()
        result['x'] = dict()
        result['dx'] = dict()
        result['da'] = dict()
        result['params'] = dict()
        result['N'] = type_num
        result['tot_num'] = np.sum(list(type_num.values()))
        result['partition'] = np.ones([result['tot_num']]).astype(np.int32)
        result['struct_type'] = structure_tags[idx]
        result['struct_weight'] = structure_weights[idx]
        result['atom_idx'] = atom_type_idx
        return result

    # Set mpi number
    def _set_mpi(self, type_num, jtem, comm):
        q = type_num[jtem] // comm.size
        r = type_num[jtem] %  comm.size

        begin = comm.rank * q + min(comm.rank, r)
        end = begin + q
        if r > comm.rank:
            end += 1
        return begin, end
        
    # Get data to make C array from variables
    def _init_sf_variables(self, type_idx, jtem, symf_params_set, atom_num, mpi_range = None ):
        if mpi_range != None: # MPI calculation
            cal_atom_idx = np.asarray(type_idx[jtem][mpi_range[0]:mpi_range[1]], dtype=np.intc, order='C')
        elif mpi_range == None: # Serial calculation
            cal_atom_idx = np.asarray(type_idx[jtem], dtype=np.intc, order='C')
        cal_atom_num = len(cal_atom_idx)

        x = np.zeros([cal_atom_num, symf_params_set[jtem]['num']], dtype=np.float64, order='C')
        dx = np.zeros([cal_atom_num, symf_params_set[jtem]['num'] * atom_num * 3], dtype=np.float64, order='C')
        da = np.zeros([cal_atom_num, symf_params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C')

        da = np.zeros([cal_atom_num, symf_params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C')

        return cal_atom_idx, cal_atom_num, x, dx, da

    def _check_error(self, errnos):   
        for errno in errnos:
            if errno == 1:
                err = "Not implemented symmetry function type."
                #if comm.rank == 0:
                self.parent.logfile.write("\nError: {:}\n".format(err))
                raise NotImplementedError(err)
            elif errno == 2:
                err = "Zeta in G4/G5 must be greater or equal to 1.0."
                #if comm.rank == 0:
                self.parent.logfile.write("\nError: {:}\n".format(err))
                raise ValueError(err)
            else:
                assert errno == 0    
    
    # Set resulatant Dictionary
    def _set_result(self, result, x, dx, da,  type_num, jtem, symf_params_set, atom_num):
        if type_num[jtem] != 0:
            # IF MPI available
            #result['x'][jtem] = np.array(comm.gather(x, root=0))
            #result['dx'][jtem] = np.array(comm.gather(dx, root=0))
            #result['da'][jtem] = np.array(comm.gather(da, root=0))
            # For Serial
            result['x'][jtem] = np.array(x)
            result['dx'][jtem] = np.array(dx)
            result['da'][jtem] = np.array(da)
            #if comm.rank == 0:
            result['x'][jtem] = np.concatenate(result['x'][jtem], axis=0).\
                                reshape([type_num[jtem], symf_params_set[jtem]['num']])
            result['dx'][jtem] = np.concatenate(result['dx'][jtem], axis=0).\
                                reshape([type_num[jtem], symf_params_set[jtem]['num'], atom_num, 3])
            result['da'][jtem] = np.concatenate(result['da'][jtem], axis=0).\
                                reshape([type_num[jtem], symf_params_set[jtem]['num'], 3, 6])
            result['partition_'+jtem] = np.ones([type_num[jtem]]).astype(np.int32)
        else:
            result['x'][jtem] = np.zeros([0, symf_params_set[jtem]['num']])
            result['dx'][jtem] = np.zeros([0, symf_params_set[jtem]['num'], atom_num, 3])
            result['da'][jtem] = np.zeros([0, symf_params_set[jtem]['num'], 3, 6])
            result['partition_'+jtem] = np.ones([0]).astype(np.int32)
        result['params'][jtem] = symf_params_set[jtem]['total']
    
    # Check ase version, E, F, S extract from snapshot, Raise Error 
    def _extract_EFS(self, result, snapshot):
        if not (self.inputs['refdata_format']=='vasp' or self.inputs['refdata_format']=='vasp-xdatcar'):
            if ase.__version__ >= '3.18.0':
                E = snapshot.get_potential_energy(force_consistent=True)
            else:
                E = snapshot.get_total_energy()
            try:
                F = snapshot.get_forces()
            except:
                if self.parent.inputs['neural_network']['use_force']:
                    err = "There is not force information! Set 'use_force' = false"
                    #if comm.rank == 0:
                    self.parent.logfile.write("\nError: {:}\n".format(err))
                    raise NotImplementedError(err)
            try:
                S = -snapshot.get_stress()/units.GPa*10
                # ASE returns the stress tensor by voigt order xx yy zz yz zx xy
                S = S[[0, 1, 2, 5, 3, 4]]
            except:
                if self.parent.inputs['neural_network']['use_stress']:
                    err = "There is not stress information! Set 'use_stress' = false"
                    #if comm.rank == 0:
                    self.parent.logfile.write("\nError: {:}\n".format(err))
                    raise NotImplementedError(err)
        return E, F, S
   
    
