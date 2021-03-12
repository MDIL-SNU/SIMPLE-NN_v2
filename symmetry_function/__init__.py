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
from ...utils.mpiclass import DummyMPI, MPI4PY
from braceexpand import braceexpand
from sklearn.decomposition import PCA

from DataGenerator import DataGenerator

def _read_params(filename):
    params_i = list()
    params_d = list()
    with open(filename, 'r') as fil:
        for line in fil:
            tmp = line.split()
            params_i += [list(map(int,   tmp[:3]))]
            params_d += [list(map(float, tmp[3:]))]

    params_i = np.asarray(params_i, dtype=np.intc, order='C')
    params_d = np.asarray(params_d, dtype=np.float64, order='C')

    return params_i, params_d


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
                                  }
                              }
        self.structure_list = './str_list'   # only use in generator
        self.pickle_list = './pickle_list'   # generator, preprocess
        self.train_data_list = './train_list'  # only preprocess
        self.valid_data_list = './valid_list'  # only preprocess
        self.comm = None
    
    def set_inputs(self):
        self.inputs = self.parent.inputs['symmetry_function']

    def generate(self):
        # Create DataGenerator and initialize
        # DataGenerator object handles [str_list], OUTCAR files, pickle files
        data_generator = DataGenerator(self.structure_list, self.pickle_list)

        # Get structure list from [str_list] file
        structures, structure_idx, structure_names, structure_weights = data_generator.parse_structure_list()







        # Get symmetry function parameter list for each atom types
        symf_params_set = dict()
        for element in self.parent.inputs['atom_types']:
            symf_params_set[element] = dict()
            symf_params_set[element]['i'], symf_params_set[element]['d'] = \
                _read_params(self.inputs['params'][element])
            symf_params_set[element]['ip'] = _gen_2Darray_for_ffi(symf_params_set[element]['i'], ffi, "int")
            symf_params_set[element]['dp'] = _gen_2Darray_for_ffi(symf_params_set[element]['d'], ffi)
            symf_params_set[element]['total'] = np.concatenate((symf_params_set[element]['i'], symf_params_set[element]['d']), axis=1)
            symf_params_set[element]['num'] = len(symf_params_set[element]['total'])
            
        for item, idx in zip(structures, structure_idx):
            # FIXME: add another input type
            if len(item) == 1:
                index = 0
                if comm.rank == 0:
                    self.parent.logfile.write('{} 0'.format(item[0]))
            else:
                if ':' in item[1]:
                    index = item[1]
                else:
                    index = int(item[1])

                if comm.rank == 0:
                    self.parent.logfile.write('{} {}'.format(item[0], item[1]))
            






            
            snapshots = data_generator.load_snapshots(self.inputs, item, index)










            for atoms in snapshots:
                cart = np.copy(atoms.get_positions(wrap=True), order='C')
                scale = np.copy(atoms.get_scaled_positions(), order='C')
                cell = np.copy(atoms.cell, order='C')

                symbols = np.array(atoms.get_chemical_symbols())
                atom_num = len(symbols)
                atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
                type_num = dict()
                type_idx = dict()
                for j,jtem in enumerate(self.parent.inputs['atom_types']):
                    tmp = symbols==jtem
                    atom_i[tmp] = j+1
                    type_num[jtem] = np.sum(tmp).astype(np.int64)
                    # if atom indexs are sorted by atom type,
                    # indexs are sorted in this part.
                    # if not, it could generate bug in training process for force training
                    type_idx[jtem] = np.arange(atom_num)[tmp]
                atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

                res = dict()
                res['x'] = dict()
                res['dx'] = dict()
                res['da'] = dict()
                res['params'] = dict()
                res['N'] = type_num
                res['tot_num'] = np.sum(list(type_num.values()))
                res['partition'] = np.ones([res['tot_num']]).astype(np.int32)
                res['struct_type'] = structure_names[idx]
                res['struct_weight'] = structure_weights[idx]
                res['atom_idx'] = atom_i

                cart_p  = _gen_2Darray_for_ffi(cart, ffi)
                scale_p = _gen_2Darray_for_ffi(scale, ffi)
                cell_p  = _gen_2Darray_for_ffi(cell, ffi)
            
                for j,jtem in enumerate(self.parent.inputs['atom_types']):
                    q = type_num[jtem] // comm.size
                    r = type_num[jtem] %  comm.size

                    begin = comm.rank * q + min(comm.rank, r)
                    end = begin + q
                    if r > comm.rank:
                        end += 1

                    cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
                    cal_num = len(cal_atoms)
                    cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

                    x = np.zeros([cal_num, symf_params_set[jtem]['num']], dtype=np.float64, order='C')
                    dx = np.zeros([cal_num, symf_params_set[jtem]['num'] * atom_num * 3], dtype=np.float64, order='C')
                    da = np.zeros([cal_num, symf_params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C')

                    x_p = _gen_2Darray_for_ffi(x, ffi)
                    dx_p = _gen_2Darray_for_ffi(dx, ffi)
                    da_p = _gen_2Darray_for_ffi(da, ffi)

                    errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                                     atom_i_p, atom_num, cal_atoms_p, cal_num, \
                                     symf_params_set[jtem]['ip'], symf_params_set[jtem]['dp'], symf_params_set[jtem]['num'], \
                                     x_p, dx_p, da_p)
                    comm.barrier()
                    errnos = comm.gather(errno)
                    errnos = comm.bcast(errnos)
                    for errno in errnos:
                        if errno == 1:
                            err = "Not implemented symmetry function type."
                            if comm.rank == 0:
                                self.parent.logfile.write("\nError: {:}\n".format(err))
                            raise NotImplementedError(err)
                        elif errno == 2:
                            err = "Zeta in G4/G5 must be greater or equal to 1.0."
                            if comm.rank == 0:
                                self.parent.logfile.write("\nError: {:}\n".format(err))
                            raise ValueError(err)
                        else:
                            assert errno == 0

                    if type_num[jtem] != 0:
                        res['x'][jtem] = np.array(comm.gather(x, root=0))
                        res['dx'][jtem] = np.array(comm.gather(dx, root=0))
                        res['da'][jtem] = np.array(comm.gather(da, root=0))
                        if comm.rank == 0:
                            res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).\
                                                reshape([type_num[jtem], symf_params_set[jtem]['num']])
                            res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
                                                reshape([type_num[jtem], symf_params_set[jtem]['num'], atom_num, 3])
                            res['da'][jtem] = np.concatenate(res['da'][jtem], axis=0).\
                                                reshape([type_num[jtem], symf_params_set[jtem]['num'], 3, 6])
                            res['partition_'+jtem] = np.ones([type_num[jtem]]).astype(np.int32)
                    else:
                        res['x'][jtem] = np.zeros([0, symf_params_set[jtem]['num']])
                        res['dx'][jtem] = np.zeros([0, symf_params_set[jtem]['num'], atom_num, 3])
                        res['da'][jtem] = np.zeros([0, symf_params_set[jtem]['num'], 3, 6])
                        res['partition_'+jtem] = np.ones([0]).astype(np.int32)
                    res['params'][jtem] = symf_params_set[jtem]['total']

                if not (self.inputs['refdata_format']=='vasp' or self.inputs['refdata_format']=='vasp-xdatcar'):
                    if ase.__version__ >= '3.18.0':
                        res['E'] = atoms.get_potential_energy(force_consistent=True)
                    else:
                        res['E'] = atoms.get_total_energy()
                    try:
                        res['F'] = atoms.get_forces()
                    except:
                        if self.parent.inputs['neural_network']['use_force']:
                            err = "There is not force information! Set 'use_force' = false"
                            if comm.rank == 0:
                                self.parent.logfile.write("\nError: {:}\n".format(err))
                            raise NotImplementedError(err)
                    try:
                        res['S'] = -atoms.get_stress()/units.GPa*10
                        # ASE returns the stress tensor by voigt order xx yy zz yz zx xy
                        res['S'] = res['S'][[0, 1, 2, 5, 3, 4]]
                    except:
                        if self.parent.inputs['neural_network']['use_stress']:
                            err = "There is not stress information! Set 'use_stress' = false"
                            if comm.rank == 0:
                                self.parent.logfile.write("\nError: {:}\n".format(err))
                            raise NotImplementedError(err)
                






                # Save "res" data to pickle file
                data_generator.save_to_pickle(res, idx)








            self.parent.logfile.write(': ~{}\n'.format(tmp_endfile))
   
    