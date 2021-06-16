import numpy as np

def _parse_symmetry_function_parameters(inputs, atom_types):
    symf_params_set = dict()
    for element in atom_types:
        symf_params_set[element] = dict()
        symf_params_set[element]['int'], symf_params_set[element]['double'] = \
            _read_params(inputs['descriptor']['params'][element])
        symf_params_set[element]['total'] = np.concatenate((symf_params_set[element]['int'], symf_params_set[element]['double']), axis=1)
        symf_params_set[element]['num'] = len(symf_params_set[element]['total'])            
    return symf_params_set

def _read_params(filename):
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
