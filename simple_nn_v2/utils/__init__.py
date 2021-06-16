from __future__ import print_function
import six
from six.moves import cPickle as pickle
import numpy as np
#from ._libgdf import lib, ffi
import os, sys, psutil, shutil
import types
import re
import collections
from collections import OrderedDict
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import array_ops, control_flow_ops, tensor_array_ops
from .mpiclass import DummyMPI, MPI4PY
from scipy.integrate import nquad
import ase
import torch
from ase.geometry import get_distances



def _generate_gdf_file(ref_list, scale, atom_types, idx_list, target_list=None, filename=None, noscale=False, sigma=0.02, comm=DummyMPI()):
    gdf = dict()
    auto_c = dict()
    auto_sigma = dict()

    for item in atom_types:
        if len(ref_list[item]) > 0:
            scaled_ref = ref_list[item] - scale[item][0:1,:]
            scaled_ref /= scale[item][1:2,:]
            scaled_ref_p = _gen_2Darray_for_ffi(scaled_ref, ffi)

            if target_list == None:
                scaled_target = scaled_ref
                scaled_target_p = scaled_ref_p
            else:
                scaled_target = target_list[item] - scale[item][0:1,:]
                scaled_target /= scale[item][1:2,:]
                scaled_target_p = _gen_2Darray_for_ffi(scaled_target, ffi)

            local_temp_gdf = np.zeros([scaled_target.shape[0]], dtype=np.float64, order='C')
            local_temp_gdf_p = ffi.cast("double *", local_temp_gdf.ctypes.data)

            if sigma == 'Auto':
                #if target_list != None:
                #    raise NotImplementedError
                #else:
                lib.calculate_gdf(scaled_ref_p, scaled_ref.shape[0], scaled_target_p, scaled_target.shape[0], scaled_ref.shape[1], -1., local_temp_gdf_p)
                local_auto_sigma = max(np.sort(local_temp_gdf))/3.
                comm.barrier()
                auto_sigma[item] = comm.allreduce_max(local_auto_sigma)

            elif isinstance(sigma, collections.Mapping):
                auto_sigma[item] = sigma[item]
            else:
                auto_sigma[item] = sigma

            lib.calculate_gdf(scaled_ref_p, scaled_ref.shape[0], scaled_target_p, scaled_target.shape[0], scaled_ref.shape[1], auto_sigma[item], local_temp_gdf_p)
            comm.barrier()

            temp_gdf = comm.gather(local_temp_gdf.reshape([-1,1]), root=0)
            comm_idx_list = comm.gather(idx_list[item].reshape([-1,1]), root=0)   

            if comm.rank == 0:
                temp_gdf = np.concatenate(temp_gdf, axis=0).reshape([-1])
                comm_idx_list = np.concatenate(comm_idx_list, axis=0).reshape([-1])

                gdf[item] = np.squeeze(np.dstack(([temp_gdf, comm_idx_list])))
                #print(gdf[item])
                gdf[item][:,0] *= float(len(gdf[item][:,0]))

                sorted_gdf = np.sort(gdf[item][:,0])
                max_line_idx = int(sorted_gdf.shape[0]*0.75)
                pfit = np.polyfit(np.arange(max_line_idx), sorted_gdf[:max_line_idx], 1)
                #auto_c[item] = np.poly1d(pfit)(sorted_gdf.shape[0]-1)
                auto_c[item] = np.poly1d(pfit)(max_line_idx-1)
                # FIXME: After testing, this part needs to be moved to neural_network.py

            """
            if callable(modifier[item]):
                gdf[item] = modifier[item](gdf[item])

            if not noscale:
                gdf[item][:,0] /= np.mean(gdf[item][:,0])
            """

    if (filename != None) and (comm.rank == 0):
        with open(filename, 'wb') as fil:
            pickle.dump(gdf, fil, protocol=2)

    return gdf, auto_sigma, auto_c

def modified_sigmoid(gdf, b=150.0, c=1.0, module_type=None):
    """
    modified sigmoid function for GDF calculation.

    :param gdf: numpy array, calculated gdf value
    :param b: float or double, coefficient for modified sigmoid
    :param c: float or double, coefficient for modified sigmoid
    """
    if module_type is None:
        module_type = np

    gdf = gdf / (1.0 + module_type.exp(-b * (gdf - c)))
    #gdf[:,0] = gdf[:,0] / (1.0 + np.exp(-b * gdf[:,0] + c))
    return gdf


def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]
    print('memory_use:', memory_use)


def read_lammps_potential(filename):
    def _read_until(fil, stop_tag):
        while True:
            line = fil.readline()
            if stop_tag in line:
                break

        return line

    shutil.copy2(filename, 'potential_read')

    weights = dict()
    with open(filename) as fil:
        atom_types = fil.readline().replace('\n','').split()[1:]
        for item in atom_types:
            weights[item] = list()            

            dims = list()
            dims.append(int(_read_until(fil, 'SYM').split()[1]))

            hidden_to_out = map(lambda x: int(x), _read_until(fil, 'NET').split()[2:])
            dims += hidden_to_out

            num_weights = len(dims) - 1

            for j in range(num_weights):
                tmp_weights = np.zeros([dims[j], dims[j+1]])
                tmp_bias = np.zeros([dims[j+1]])

                # Since PCA will be dealt separately, skip PCA layer.
                skip = True if fil.readline().split()[-1] == 'PCA' else False
                for k in range(dims[j+1]):
                    tmp_weights[:,k] = list(map(lambda x: float(x), fil.readline().split()[1:]))
                    tmp_bias[k] = float(fil.readline().split()[1])

                if skip:
                    continue
                weights[item].append(np.copy(tmp_weights))
                weights[item].append(np.copy(tmp_bias))

    return weights
