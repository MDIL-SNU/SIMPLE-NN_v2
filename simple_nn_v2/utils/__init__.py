from __future__ import print_function
import six
import numpy as np
import torch
import os, sys, psutil, shutil
import types, re, collections

import ase
import torch
from ase.geometry import get_distances

from simple_nn_v2.utils.features import _gen_2Darray_for_ffi
from simple_nn_v2.features.mpi import DummyMPI, MPI4PY
from ._libgdf import lib, ffi


def _generate_gdf_file(ref_list, scale, atom_types, idx_list, target_list=None, sigma=0.02, comm=DummyMPI()):
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
                local_auto_sigma = max(np.sort(local_temp_gdf)) / 3.
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
                gdf[item][:,0] *= float(len(gdf[item][:,0]))

                sorted_gdf = np.sort(gdf[item][:,0])
                max_line_idx = int(sorted_gdf.shape[0]*0.75)
                pfit = np.polyfit(np.arange(max_line_idx), sorted_gdf[:max_line_idx], 1)
                #auto_c[item] = np.poly1d(pfit)(sorted_gdf.shape[0]-1)
                auto_c[item] = np.poly1d(pfit)(max_line_idx-1)
                # FIXME: After testing, this part needs to be moved to neural_network.py


    return gdf, auto_sigma, auto_c

def modified_sigmoid(gdf, b=150.0, c=1.0, module_type=None):
    """
    modified sigmoid function for GDF calculation.

    :param gdf: numpy array, calculated gdf value
    :param b: float or double, coefficient for modified sigmoid
    :param c: float or double, coefficient for modified sigmoid
    """
    if module_type is None:
        module_type = torch
    #elif module_type is 'torch':
    #    module_type = torch

    gdf = gdf / (1.0 + module_type.exp(-b * (gdf - c)))
    #gdf[:,0] = gdf[:,0] / (1.0 + np.exp(-b * gdf[:,0] + c))
    return gdf


def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]
    print('memory_use:', memory_use)
