import sys
sys.path.append('../../../')

from simple_nn_v2 import Simple_nn
from simple_nn_v2.features.symmetry_function import Symmetry_function

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
model = Simple_nn('input.yaml', descriptor=Symmetry_function())
descriptor = Symmetry_function()
descriptor.parent = model
descriptor.set_inputs()

""" Previous setting before test code

1. load snapshot from FILE
2. extract type_num, atom_idx from _get_structure_info()
3. set example structure_tags, structure_weights, idx

"""
# 1. load snapshot
from ase import io
FILE = '../../test_data/SiO2/OUTCAR_comp'
snapshots = io.read(FILE, index=':2:', format='vasp-out')
snapshot = snapshots[0]

# 2. extract from _get_structure_info()
atom_num, atom_type_idx, type_num, type_atom_idx, cart, scale, cell = descriptor._get_structure_info(snapshot)

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
result = descriptor._init_result(type_num, structure_tags, structure_weights, idx, atom_type_idx)

print("1. chcek if 'x', 'dx', 'da', 'params' is empty dictionary")
print 'x: ',result['x']
print 'dx: ', result['dx']
print 'da: ', result['da']
print 'params: ', result['params']

print("\n2. check if 'N' has correct element types and atom number for each elements")
print 'N: ', result['N']

print("\n3. check if 'tot_num' has total atom number")
print 'tot_num: ', result['tot_num']
#print 'partition: ', result['partition']

print("\n4. check if 'struct_type', 'struct_weight' has correct tag, weight with correspond to idx (we set in previous setting # 3)")
print'For structure tag: ', structure_tags, ', structure weights: ', structure_weights, ', index: %s'%idx
print 'struct_type: ', result['struct_type']
print 'struct_weight: ', result['struct_weight']

print("\n5. check if 'atom_idx' set correctly")
prev=0
end=0
for elem in result['N']:
    end += result['N'][elem]
    print 'result["N"][%s] atom_idx: '%elem, result['atom_idx'][prev:end], len(result['atom_idx'][prev:end])
    prev += result['N'][elem]
