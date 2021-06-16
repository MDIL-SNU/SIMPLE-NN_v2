import numpy as np
from simple_nn_v2.features.symmetry_function import utils as utils_symf

def get_scale_function(scale_type='minmax'):
    scale_function = {
        'minmax': minmax,
        'meanstd': meanstd,
        'uniform_gas': uniform_gas
    }
    return scale_function[scale_type]

def minmax(feature_list, atom_type, scale_scale):
    mid_range = 0.5 * (np.amax(feature_list[atom_type], axis=0) + np.amin(feature_list[atom_type], axis=0))
    width = 0.5 * (np.amax(feature_list[atom_type], axis=0) - np.amin(feature_list[atom_type], axis=0)) / scale_scale
    return mid_range, width

def meanstd(feature_list, atom_type, scale_scale):
    mean = np.mean(feature_list[atom_type], axis=0)
    std_dev = np.std(feature_list[atom_type], axis=0) / scale_scale
    return mean, std_dev

def uniform_gas(feature_list, atom_type, scale_scale):
    params_set = dict()
    params_set['int'], params_set['double'] = utils_symf._read_params(inputs['descriptor']['params'][atom_type])

    assert params is not None and scale_rho is not None
    mean = np.mean(feature_list[atom_type], axis=0)
    inp_size = feature_list[atom_type].shape[1]
    uniform_gas_scale = np.zeros(inp_size)

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

    for p in range(inp_size):
        if params['int'][p,0] == 2:
            ti = atom_types[params['int'][p,1] - 1]
            eta = params['double'][p,1]
            rc = params['double'][p,0]
            rs = params['double'][p,2]
            uniform_gas_scale[p] = scale_rho[ti] * nquad(G2, [[0,rc]])[0]
        elif params['int'][p,0] == 4:
            ti = atom_types[params['int'][p,1] - 1]
            tj = atom_types[params['int'][p,2] - 1]
            eta = params['double'][p,1]
            rc = params['double'][p,0]
            zeta = params['double'][p,2]
            lamb = params['double'][p,3]
            uniform_gas_scale[p] = scale_rho[ti] * scale_rho[tj] * 4 * np.pi *\
                                (nquad(G4, [[0,rc], [0,rc], [0, np.pi]])[0] -\
                                (nquad(singular, [[0,rc]])[0] if lamb == 1 else 0))
        elif params['int'][p,0] == 5:
            ti = atom_types[params['int'][p,1] - 1]
            tj = atom_types[params['int'][p,2] - 1]
            eta = params['double'][p,1]
            rc = params['double'][p,0]
            zeta = params['double'][p,2]
            lamb = params['double'][p,3]
            uniform_gas_scale[1] = scale_rho[ti] * scale_rho[tj] * 4 * np.pi *\
                                (nquad(G5, [[0,rc], [0,rc], [0, np.pi]])[0] -\
                                (nquad(singular, [[0,rc]])[0] if lamb == 1 else 0))
        else:
            assert False

    return mean, uniform_gas_scale
