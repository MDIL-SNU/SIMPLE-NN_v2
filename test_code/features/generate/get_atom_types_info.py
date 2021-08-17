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

""" Previous setting before test code

    1. load structure from FILE

"""
from ase import io
########## Set This Variable ###########
#FILE = './OUTCAR_comp'
FILE =rootdir+'OUTCAR_SiO_comp'
########################################
structures = io.read(FILE, index='::', format='vasp-out')#, force_consistent=True)
structure = structures[0]

""" Main test code

    Test _get_structure_info()
    1. check if "atom_num" is total atom number
    2. check "type_num" has correct element types and atom number for each elements
    3. check if atom_type_idx has correct values
    4. check "type_atom_idx" has correct atom index 
    5. check lattice parameter
    6. check Cartesian coordination in "cart" (first 5, last 5 coordination)
    7. check Fractional coordination in "scale" (first 5, last 5 coordination)

"""
import numpy as np

atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type = generating._get_atom_types_info(structure, atom_types)

print('1. check if "atom_num" is total atom number')
print('atom num: %s\n'%atom_num)
if atom_num != structure.get_global_number_of_atoms():
    print("Error occured : different value for total atom number {0} {1}. aborting.".format(atom_num,structure.get_global_number_of_atoms()))
    print(sys.exc_info())
    os.abort()

print('2. check "atoms_per_type" has correct element types and atom number for each elements')
print('atoms_per_type: %s\n'%atoms_per_type)
import re
tmp_formula =  "Si24O48" #Answer for OUTCAR_comp
type_reg = re.compile(r'[a-zA-Z]+')
number_reg = re.compile(r'\d+')
type_list = type_reg.findall(tmp_formula)
number_list = number_reg.findall(tmp_formula)

try: #Check well match with ase module
    for idx, item in enumerate(atoms_per_type.keys()):
        if item in atoms_per_type: #key exist check
            assert atoms_per_type[item] == int(number_list[idx])
        else:
            raise Exception(f"No atom type ({item}) in atoms_per_type.")
except AssertionError:
    print("Error occured : different value for atoms_per_type: {0} with {1} {2}. aborting".format(item,atoms_per_type[item],number_list[idx]))
    print(sys.exc_info())
    os.abort()
except:
    print("Error occured. aborting")
    print(sys.exc_info())
    os.abort()



print('3. check if "atom_type_idx" is total atom number')
print('atom_type_idx: %s\n'%atom_type_idx)
#Use ase.get_atomic_numbers to check 
tmp_type_idx = structure.get_atomic_numbers()
number_dict  = {'Si':14,'O': 8} #Atomic type to numbe
index_dict   = {1:'Si', 2:'O'} #Atomic index to type
try:
    for idx in range(len(atom_type_idx)):
        assert number_dict[index_dict[atom_type_idx[idx]]] == int(tmp_type_idx[idx])
except:
    print(f"Error occured : wrong atom type index matching: {idx}")
    print(sys.exc_info())
    os.abort()


print('4. check if "atom_idx_per_type" is total atom number')
print('atom_idx_per_type: %s\n'%atom_idx_per_type)
try:
    tmp_idx = 0
    for idx, atom_type in enumerate(type_list):
        assert atom_idx_per_type[atom_type][-1] == (int(number_list[idx])-1+tmp_idx)
        tmp_idx += int(number_list[idx])
except:
    print(f"Error occured : wrong atom_idx_per_type : {atom_type} ")
    print(sys.exc_info())
    os.abort()


