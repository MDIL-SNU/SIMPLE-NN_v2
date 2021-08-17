import sys
#sys.path.append('../../../../../')
sys.path.append('./')
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating
from simple_nn_v2.features.symmetry_function import utils as symf_utils

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
default='./input.yaml'
pytest ='./test_input/input_SiO.yaml'
yaml = pytest


logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(yaml, logfile)
atom_types = inputs['atom_types']

""" Main test code

Test _parsing_symf_params()
1. Check if 'num' for each elements
2. Check key 'total', 'int', 'double' values for each elements

"""
symf_params_set = symf_utils._parse_symmetry_function_parameters(inputs, atom_types)
print("Check symf_params_set keys")
print(symf_params_set.keys())
try:
    for elem in symf_params_set.keys():
        print("Check symf_params_set[%s]"%elem)
        print("keys: ", symf_params_set[elem].keys())
        print("['num']: ", symf_params_set[elem]['num'])

        # Check ['total'], ['int'], ['double'] values
        f=open(inputs['descriptor']['params'][elem],'r')
        lines=f.readlines()
        f.close()

    for i, line in enumerate(lines):
        vals = line.split()
        for j in range(len(vals)):
            assert float(vals[j]) == symf_params_set[elem]['total'][i][j],\
            f'ValueError in key: total elem: {elem}  {i+1} th symf, {j+1} th value'
            if j<3:
                assert float(vals[j]) == symf_params_set[elem]['int'][i][j],\
                    f'ValueError in key: i  elem: {elem}  {i+1} th symf, {j+1} th value'
            elif j>=3:
                assert float(vals[j]) == symf_params_set[elem]['double'][i][j-3],\
                    f'ValueError in key: d  elem: {elem}  {i+1} th symf, {j+1} th value'
    print("All parse passed.")
except AssertionError:
    print(sys.exc_info())
    print("Error occred : parse_symmetry_function_parameter")
    os.abort()
