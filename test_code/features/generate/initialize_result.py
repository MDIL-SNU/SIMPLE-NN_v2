import os
import sys
#sys.path.append('../../../../../')
sys.path.append('./')

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
#rootdir = './'
rootdir = './test_input/'
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)
atom_types = inputs['atom_types']
inputs = inputs['descriptor']

""" Previous setting before test code

1. load structure from FILE
2. extract atoms_per_type, atom_idx from _get_structure_info()
3. set example structure_tags, structure_weights, idx

"""
# 1. load structure
from ase import io
FILE = rootdir+'OUTCAR_SiO_comp'
structures = io.read(FILE, index=':2:', format='vasp-out')
structure = structures[0]

# 2. extract from _get_structure_info()
cell, cart, scale = generating._get_structure_coordination_info(structure)
atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type = generating._get_atom_types_info(structure, atom_types)

# 3. set example variables
structure_tags = ['None', 'Data1', 'Data2', 'Data3']
structure_weights = [1, 1, 3, 3]
idx=2

""" Main test code

    Test _init_result()
   1. chcek if 'x', 'dx', 'da', 'params' is empty dictionary
   2. check if 'N' has correct element types and atom number for each elements
   3. check if 'tot_num' has total atom number
   4. check if 'struct_type', 'struct_weight' has correct tag, weight with correspond to idx
   5. check if 'atom_idx' set correctly

"""
result = generating._initialize_result(atoms_per_type, structure_tags, structure_weights, idx, atom_type_idx)
check_list = ['x','dx','da','N','tot_num','struct_type','struct_weight']



print("1. chcek if 'x', 'dx', 'da', 'params' is empty dictionary")
for idx in range(3):
    if not result[check_list[idx]]:
        print(f'{idx} is empty : ', result[check_list[idx]])
    else:
        print(f'{idx} is not  empty : ', result[check_list[idx]] , ' Aborting.')
        os.abort()

print("\n2. check if 'N' has correct element types and atom number for each elements")
print('N: ', result['N'])
#Correct answer
ans_dict = {'Si':24, 'O':48}
try:
    if result['N'].keys() == ans_dict.keys():
        for atype in list(ans_dict.keys()):
            assert ans_dict[atype] == result['N'][atype]
    else:
        raise Exception("Not same keys in result")
except AssertionError:
    print(f"Error occured : not same atom number in initialize_result : {atype}")
    print(sys.exc_info())
    os.abort()
except:
    print(f"Error occured: Aborting")
    print(sys.exc_info())
    os.abort()
    
print("\n3. check if 'tot_num' has total atom number")
print('tot_num: ', result['tot_num'])
#Totam number 72
try:
    assert result['tot_num'] == 72
except AssertionError:
    print("Error occured : wrong total number in initialize_result : {0}, {1}".format(result['tot_num'],72))
    print(sys.exc_info())
    os.abort()

#?
print("\n4. check if 'struct_type', 'struct_weight' has correct tag, weight with correspond to idx(we set in previous setting # 3)")
print('For structure tag: ', structure_tags, ', structure weights: ', structure_weights, ', index: %s'%idx)
print('struct_type: ', result['struct_type'])
print('struct_weight: ', result['struct_weight'])

print("\n5. check if 'atom_idx' set correctly")
prev=0
end=0
for elem in result['N']:
    end += result['N'][elem]
    print('result["N"][%s] atom_idx: '%elem, result['atom_idx'][prev:end], len(result['atom_idx'][prev:end]))
    prev += result['N'][elem]

idx_dict = {'Si':1, 'O':2}
tmp_idx = 0
try:
    for atype in list(idx_dict.keys()):
        for idx in range(result['N'][atype]):
            assert idx_dict[atype] == int(result['atom_idx'][tmp_idx+idx])
        tmp_idx += result['N'][atype]
except AssertionError:
    print("Error occured : incorrect atom inddex in initialize_result")
    print(sys.exc_info())
    os.abort()
