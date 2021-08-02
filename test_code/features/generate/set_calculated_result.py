import sys
#sys.path.append('../../../../../')
sys.path.append('./')
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating
from simple_nn_v2.features.symmetry_function import utils as symf_utils
from simple_nn_v2.features.symmetry_function._libsymf import lib, ffi
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features.symmetry_function.mpi import DummyMPI


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
    4. initialize result and symmetry function variables

"""
# 1. parsing params_XX
symf_params_set = symf_utils._parse_symmetry_function_parameters(inputs, atom_types)
for element in atom_types:
    symf_params_set[element]['int_p'] = util_ft._gen_2Darray_for_ffi(symf_params_set[element]['int'], ffi, "int")
    symf_params_set[element]['double_p'] = util_ft._gen_2Darray_for_ffi(symf_params_set[element]['double'], ffi)

# 2. load structure from FILE
from ase import io
FILE = rootdir+'OUTCAR_SiO_comp'
structures = io.read(FILE, index='::', format='vasp-out')
structure = structures[0]
structure_tags = ['None', 'Data1', 'Data2', 'Data3']
structure_weights = [1, 1, 3, 3]

# 3. extract type_idx, atom_num from _get_structure_info()
cell, cart, scale = generating._get_structure_coordination_info(structure)
atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type = generating._get_atom_types_info(structure, atom_types)
atom_type_idx_p = ffi.cast("int *", atom_type_idx.ctypes.data)
cart_p  = util_ft._gen_2Darray_for_ffi(cart, ffi)
scale_p = util_ft._gen_2Darray_for_ffi(scale, ffi)
cell_p  = util_ft._gen_2Darray_for_ffi(cell, ffi)

# 4. initialize result and symmetry function variables
idx = 1
result = generating._initialize_result(atoms_per_type, structure_tags, structure_weights, idx, atom_type_idx)
element = 'Si'
cal_atom_idx, cal_atom_num, x, dx, da = generating._initialize_symmetry_function_variables(atom_idx_per_type, element, symf_params_set, atom_num, mpi_range = None )
cal_atom_idx_p = ffi.cast("int *", cal_atom_idx.ctypes.data)
x_p = util_ft._gen_2Darray_for_ffi(x, ffi)
dx_p = util_ft._gen_2Darray_for_ffi(dx, ffi)
da_p = util_ft._gen_2Darray_for_ffi(da, ffi)

# 5. calculate symmetry function
errno = lib.calculate_sf(cell_p, cart_p, scale_p, atom_type_idx_p, atom_num,\
        cal_atom_idx_p, cal_atom_num, symf_params_set[element]['int_p'],\
        symf_params_set[element]['double_p'], symf_params_set[element]['num'],\
        x_p, dx_p, da_p)

""" Main test code

    Test _set_result()
    1. check if 'N', 'tot_num', 'struct_type', 'struct_weight' is identical to test3 results
    2. check if 'x', 'dx', 'da' has available values (didn't check if has identical values)
    3. check if 'E', 'F', 'S' value extract correctly

"""

comm = DummyMPI()
generating._set_calculated_result(inputs, result, x, dx, da, atoms_per_type, element, symf_params_set, atom_num, comm)

print(result)
print("1. check if 'N', 'tot_num', 'struct_type', 'struct_weight', 'atom_idx' are identical to test3 results")
print('N: ', result['N'])
print('tot_num: ', result['tot_num'])
print('struct_type: ', result['struct_type'])
print('struct_weight: ', result['struct_weight'])
prev=0
end=0
for elem in result['N']:
    end += result['N'][elem]
    print('result["N"][%s] atom_idx: '%elem, result['atom_idx'][prev:end], len(result['atom_idx'][prev:end]))
    prev += result['N'][elem]

print("\n2. check if 'x', 'dx', 'da' has available values (didn't check if has identical values)")
print(' x: ', result['x']['Si'])
print('dx: ', result['dx']['Si'][0])
print('da: ', result['da']['Si'][0])


print("\n3. check if 'E', 'F', 'S' value extract correctly")
E, F, S = generating._extract_EFS(inputs, structure, logfile, comm)
print('E= %s'%E)
for i in range(len(F)):
    if i==0:
        print('F= %s'%F[i])
    elif i<5 or i>len(F)-5:
        print(F[i])
    elif i==5:
        print('...')
print('S= %s'%S)
