import sys
sys.path.append('../')

from simple_nn_v2 import Simple_nn
from simple_nn_v2.features.symmetry_function import Symmetry_function
from simple_nn_v2.features.symmetry_function._libsymf import lib, ffi
from simple_nn_v2.utils import _gen_2Darray_for_ffi

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

model = Simple_nn('input.yaml', descriptor=Symmetry_function())
descriptor = Symmetry_function()
descriptor.parent = model
descriptor.set_inputs()

""" Previous setting before test code

    1. parsing params_XX
    2. load snapshot from FILE
    3. extract type_idx, atom_num from _get_structure_info()
    4. initialize result and symmetry function variables

"""
# 1. parsing params_XX
symf_params_set = descriptor._parsing_symf_params()
for element in model.inputs['atom_types']:
    symf_params_set[element]['int_p'] = _gen_2Darray_for_ffi(symf_params_set[element]['int'], ffi, "int")
    symf_params_set[element]['double_p'] = _gen_2Darray_for_ffi(symf_params_set[element]['double'], ffi)

# 2. load snapshot from FILE
from ase import io
FILE = '../test_data/generate/OUTCAR_2'
snapshots = io.read(FILE, index='::', format='vasp-out')
snapshot = snapshots[0]
structure_tags = ['None', 'Data1', 'Data2', 'Data3']
structure_weights = [1, 1, 3, 3]

# 3. extract type_idx, atom_num from _get_structure_info()
atom_num, atom_type_idx, type_num, type_atom_idx, cart, scale, cell = descriptor._get_structure_info(snapshot)
atom_type_idx_p = ffi.cast("int *", atom_type_idx.ctypes.data)
cart_p  = _gen_2Darray_for_ffi(cart, ffi)
scale_p = _gen_2Darray_for_ffi(scale, ffi)
cell_p  = _gen_2Darray_for_ffi(cell, ffi)

# 4. initialize result and symmetry function variables
idx = 1
result = descriptor._init_result(type_num, structure_tags, structure_weights, idx, atom_type_idx)
jtem = 'Sb'
cal_atom_idx, cal_atom_num, x, dx, da = descriptor._init_sf_variables(type_atom_idx, jtem, symf_params_set, atom_num, mpi_range = None )
cal_atom_idx_p = ffi.cast("int *", cal_atom_idx.ctypes.data)
x_p = _gen_2Darray_for_ffi(x, ffi)
dx_p = _gen_2Darray_for_ffi(dx, ffi)
da_p = _gen_2Darray_for_ffi(da, ffi)

# 5. calculate symmetry function
errno = lib.calculate_sf(cell_p, cart_p, scale_p, atom_type_idx_p, atom_num,\
        cal_atom_idx_p, cal_atom_num, symf_params_set[jtem]['int_p'],\
        symf_params_set[jtem]['double_p'], symf_params_set[jtem]['num'],\
        x_p, dx_p, da_p)

""" Main test code

    Test _set_result()
    1. check if 'N', 'tot_num', 'struct_type', 'struct_weight' is identical to test3 results
    2. check if 'x', 'dx', 'da' has available values (didn't check if has identical values)
    3. check if 'E', 'F', 'S' value extract correctly

"""
descriptor._set_result(result, x, dx, da, type_num, jtem, symf_params_set, atom_num)

print(result)
print("1. check if 'N', 'tot_num', 'struct_type', 'struct_weight', 'atom_idx' are identical to test3 results")
print 'N: ', result['N']
print 'tot_num: ', result['tot_num']
#print 'partition: ', result['partition']
print 'struct_type: ', result['struct_type']
print 'struct_weight: ', result['struct_weight']
prev=0
end=0
for elem in result['N']:
    end += result['N'][elem]
    print 'result["N"][%s] atom_idx: '%elem, result['atom_idx'][prev:end], len(result['atom_idx'][prev:end])
    prev += result['N'][elem]

print("\n2. check if 'x', 'dx', 'da' has available values (didn't check if has identical values)")
print 'x: ', result['x']['Sb']
print'dx: ', result['dx']['Sb'][0]
print'da: ', result['da']['Sb'][0]


print("\n3. check if 'E', 'F', 'S' value extract correctly")
E, F, S = descriptor._extract_EFS(snapshot)
print('E= %s'%E)
for i in range(len(F)):
    if i==0:
        print('F= %s'%F[i])
    elif i<5 or i>len(F)-5:
        print(F[i])
    elif i==5:
        print('...')
print('S= %s'%S)
