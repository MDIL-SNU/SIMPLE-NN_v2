import numpy as np
from simple_nn_v2.features.symmetry_function import utils as utils_symf
from scipy.integrate import nquad


def get_scale_function(scale_type='minmax'):
    scale_function = {
        'minmax': minmax,
        'meanstd': meanstd,
        'uniform gas': uniform_gas
    }
    return scale_function[scale_type]

def minmax(inputs, feature_list, atom_type, comm):
    scale_width = inputs['preprocessing']['scale_width']
    mid_range = 0.5 * (np.amax(feature_list[atom_type], axis=0) + np.amin(feature_list[atom_type], axis=0))
    width = 0.5 * (np.amax(feature_list[atom_type], axis=0) - np.amin(feature_list[atom_type], axis=0)) / scale_width
    return mid_range, width

def meanstd(inputs, feature_list, atom_type, comm):
    scale_width = inputs['preprocessing']['scale_width']
    mean = np.mean(feature_list[atom_type], axis=0)
    std_dev = np.std(feature_list[atom_type], axis=0) / scale_width
    return mean, std_dev

def uniform_gas(inputs, feature_list, atom_type, comm):
    scale_width = inputs['preprocessing']['scale_width']
    atom_types = inputs['atom_types']
    scale_rho = inputs['preprocessing']['scale_rho']
    params_set = dict()
    params_set['int'], params_set['double'] = utils_symf._read_params(inputs['params'][atom_type])

    assert params_set is not None and inputs['preprocessing']['scale_rho'] is not None
    mean = np.mean(feature_list[atom_type], axis=0)
    inp_size = feature_list[atom_type].shape[1]
    sendbuf = np.zeros(inp_size)
    
    mpi_quotient = inp_size // comm.size
    mpi_remainder = inp_size % comm.size

    begin_idx = comm.rank * mpi_quotient + min(comm.rank, mpi_remainder)
    end_idx = begin_idx + mpi_quotient
    if mpi_remainder > comm.rank:
        end_idx += 1

    count = end_idx - begin_idx
    sendbuf = np.zeros(count)
    recvbuf = np.zeros(inp_size)

    count = comm.allgather(count)
    displ = np.zeros(comm.size, dtype=np.int)
    for i in range(1, comm.size):
        displ[i] = displ[i - 1] + count[i - 1]

    def G2(r):
        return 4 * np.pi * r**2 * np.exp(-eta * (r - rs)**2) * 0.5 * (np.cos(np.pi * r / rc) + 1)

    # fix r_ij along z axis and use symmetry
    def G4(r1, r2, th2):
        r3 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(th2))
        fc3 = 0.5 * (np.cos(np.pi * r3 / rc) + 1) if r3 < rc else 0.0
        return r1**2 * r2**2 * np.sin(th2) * 2 * np.pi *\
                2**(1-zeta) * (1 + lamb * np.cos(th2))**zeta * np.exp(-eta * (r1**2 + r2**2 + r3**2)) *\
                0.5 * (np.cos(np.pi * r1 / rc) + 1) * 0.5 * (np.cos(np.pi * r2 / rc) + 1) * fc3

    def G5(r1, r2, th2):
        return r1**2 * r2**2 * np.sin(th2) * 2 * np.pi *\
                    2**(1-zeta) * (1 + lamb * np.cos(th2))**zeta * np.exp(-eta * (r1**2 + r2**2)) *\
                    0.5 * (np.cos(np.pi * r1 / rc) + 1) * 0.5 * (np.cos(np.pi * r2 / rc) + 1)

    # subtract G4 when j==k (r1==r2)
    def singular(r1):
        # r3 = 0, fc3 = 1
        return r1**4 * 2**(1-zeta) * (1 + lamb)**zeta * np.exp(-eta * 2 * r1**2) *\
                (0.5 * (np.cos(np.pi * r1 / rc) + 1))**2

    for p in range(begin_idx, end_idx):
        if params_set['int'][p,0] == 2:
            ti = atom_types[params_set['int'][p,1] - 1]
            eta = params_set['double'][p,1]
            rc = params_set['double'][p,0]
            rs = params_set['double'][p,2]
            sendbuf[p - begin_idx] = scale_rho[ti] * nquad(G2, [[0,rc]])[0]
        elif params_set['int'][p,0] == 4:
            ti = atom_types[params_set['int'][p,1] - 1]
            tj = atom_types[params_set['int'][p,2] - 1]
            eta = params_set['double'][p,1]
            rc = params_set['double'][p,0]
            zeta = params_set['double'][p,2]
            lamb = params_set['double'][p,3]
            sendbuf[p - begin_idx] = scale_rho[ti] * scale_rho[tj] * 4 * np.pi *\
                                (nquad(G4, [[0,rc], [0,rc], [0, np.pi]])[0] -\
                                (nquad(singular, [[0,rc]])[0] if lamb == 1 else 0))
        elif params_set['int'][p,0] == 5:
            ti = atom_types[params_set['int'][p,1] - 1]
            tj = atom_types[params_set['int'][p,2] - 1]
            eta = params_set['double'][p,1]
            rc = params_set['double'][p,0]
            zeta = params_set['double'][p,2]
            lamb = params_set['double'][p,3]
            sendbuf[p - begin_idx] = scale_rho[ti] * scale_rho[tj] * 4 * np.pi *\
                                (nquad(G5, [[0,rc], [0,rc], [0, np.pi]])[0] -\
                                (nquad(singular, [[0,rc]])[0] if lamb == 1 else 0))
        else:
            assert False

    comm.Allgatherv(sendbuf, recvbuf, count, displ, "double")
    return mean, recvbuf
