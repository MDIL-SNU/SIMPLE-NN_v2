import os
import sys
#sys.path.append('../../../../../')
sys.path.append('./')
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating
from simple_nn_v2.features.symmetry_function import utils as symf_utils

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
rootdir='./'
rootdir='./test_input/'
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)
atom_types = inputs['atom_types']

""" Previous setting before test code

    1. parsing params_XX
    2. load structure from FILE
    3. extract type_idx, atom_num from _get_structure_info()

"""
symf_params_set = symf_utils._parse_symmetry_function_parameters(inputs, atom_types)

from ase import io
FILE = rootdir+'OUTCAR_SiO_comp'
structures = io.read(FILE, index=':2:', format='vasp-out')
structure = structures[0]
structure_tags = ['Data1', 'Data2', 'Data3']
structure_weights = [1, 3, 3]
cell, cart, scale = generating._get_structure_coordination_info(structure)
atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type = generating._get_atom_types_info(structure, atom_types)

""" Main test code

    Test _init_sf_variables()
    1. check if "jtem" atom number is correct
    2. check if "jtem" atom idx is correct
    3. check if 'x', 'dx', 'da' is initialze to 0

"""

jtem = 'Si'
cal_atom_idx, cal_atom_num, x, dx, da = generating._initialize_symmetry_function_variables(atom_idx_per_type, jtem, symf_params_set, atom_num, mpi_range = None )

print('1. check if "jtem" atom number is correct')
print(f'{jtem} cal_num: ', cal_atom_num )
try:
    assert cal_atom_num == 24
except AssertionError:
    print(f"Error occured : {jtem} cal_atom_num wrong")
    print(sys.exc_info())
    os.abort()

print('\n2. check if "%s" atom idx is correct)'%jtem)
print('cal_atom_idx: ')
tmp_str =  ""
try:
    for i in range(cal_atom_num):
        tmp_str += " "+str(cal_atom_idx[i])
        assert i == cal_atom_idx[i]
    print(tmp_str)
except AssertionError:
    print(f"Error occured : wrong {jtem} index")
    print(sys.exc_info())
    os.abort()    

print("\n3. check if 'x', 'dx', 'da' is initialze to 0")



try:
    print("checking x")
    item = "x"
    for i in range(len(x)):
        for j in range(len(x[i])):
            assert x[i][j] == 0
    print("x initialize ok")

    print("checking dx")
    item = "dx"
    for i in range(len(dx)):
        for j in range(len(dx[i])):
            assert dx[i][j] == 0
    print("dx initialize ok")

    print("checking da")
    item = "da"
    for i in range(len(da)):
        for j in range(len(da[i])):
            assert da[i][j] == 0
    print("da initialize ok")

except AssertionError:
    print(f"Error occured : {i+1} th atom's {j+1} th {item} value initialize not correctly")
    print(sys.exc_info())
    os.abort()
