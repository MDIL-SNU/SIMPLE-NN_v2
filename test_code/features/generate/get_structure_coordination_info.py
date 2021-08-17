import os
import sys
#For pytest
#sys.path.append('../../../../../')
sys.path.append('./')
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating
import numpy as np
from ase import io

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

#root_dir = './'
root_dir = './test_input/'


logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(root_dir+'input_SiO.yaml', logfile)
atom_types = inputs['atom_types']

""" Previous setting before test code

    1. load structure from FILE

"""
########## Set This Variable ###########
FILE = root_dir+'OUTCAR_SiO_comp'
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

cell, cart, scale= generating._get_structure_coordination_info(structure)
cell_comp = np.copy(structure.cell, order='C')
cart_comp=np.copy(structure.get_positions(wrap=True), order='C')
scale_comp = np.copy(structure.get_scaled_positions(), order='C')

print('1. check lattice parameter')
print('Lattice parameter')
try:
    for i in range(3):
        print(cell[i][0], cell[i][1], cell[i][2])
        for j in range(3):
            assert cell[i][j] == cell_comp[i][j]
except AssertionError: 
    print(f"Error : {i}, {j} index lattice parameter has wrong values, aborting.")
    sys.exc_info()
    os.abort()
            

print('\n2. check Cartesian coordination in "cart" (random 10 coordinations)')
rand_idx = list()
while len(rand_idx) != 10:
    tmp_idx = np.random.randint(len(cart)) 
    if tmp_idx not in rand_idx:
        rand_idx.append(tmp_idx)
print('Generated random index : ',rand_idx)

try:
    print('Cartesian coordination')
    for idx in rand_idx:
        print(f'IDX {idx} : ',cart[idx][0], cart[idx][1], cart[idx][2])
        for xyz in range(3):
            assert cart[idx][xyz] == cart_comp[idx][xyz]
except AssertionError:
    print(f"{idx+1} th atom's {xyz} th cartesian coordination has wrong values")
    sys.exc_info()
    os.abort()

print('\n3. check Fractional coordination in "cart_p" (random 10 coordinations)')
print('Fractional coordination')
try:
    for idx in rand_idx:
        print(f'IDX {idx} : ',scale[idx][0], scale[idx][1], scale[idx][2])
        for xyz in range(3):
            assert scale[idx][xyz] == scale_comp[idx][xyz]
except AssertionError:
    print(f"Error : {idx+1} th atom's {xyz} th fractional coordination has wrong values")
    sys.exc_info()
    os.abort()

