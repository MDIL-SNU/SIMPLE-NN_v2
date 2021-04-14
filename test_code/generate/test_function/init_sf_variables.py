import sys
sys.path.append('../../../')

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('./input.yaml', logfile)
atom_types = inputs['atom_types']
inputs = inputs['symmetry_function']

""" Previous setting before test code

    1. parsing params_XX
    2. load snapshot from FILE
    3. extract type_idx, atom_num from _get_structure_info()

"""
symf_params_set = generating._parsing_symf_params(inputs, atom_types)

from ase import io
FILE = '../../test_data/SiO2/OUTCAR_comp'
snapshots = io.read(FILE, index=':2:', format='vasp-out')
snapshot = snapshots[0]
structure_tags = ['Data1', 'Data2', 'Data3']
structure_weights = [1, 3, 3]
atom_num, atom_type_idx, type_num, type_atom_idx, cart, scale, cell = generating._get_structure_info(snapshot, atom_types)


""" Main test code

    Test _init_sf_variables()
    1. check if "jtem" atom number is correct
    2. check if "jtem" atom idx is correct
    3. check if 'x', 'dx', 'da' is initialze to 0

"""
jtem = 'Si'
cal_atom_idx, cal_atom_num, x, dx, da = generating._init_sf_variables(type_atom_idx, jtem, symf_params_set, atom_num, mpi_range = None )

print('1. check if "jtem" atom number is correct')
print 'cal_num: ', cal_atom_num

print('\n2. check if "%s" atom idx is correct)'%jtem)
print('cal_atom_idx: '),
for i in range(cal_atom_num):
    print(cal_atom_idx[i]),
print

print("\n3. check if 'x', 'dx', 'da' is initialze to 0")
x_e = False
dx_e = False
da_e = False
for i in range(len(x)):
    for j in range(len(x[i])):
        if x[i][j] != 0:
            x_e = True
            print("%sth atom's %sth x value initialize not correctly"%(i+1, j+1))
if not x_e:
    print"x initialize ok"

for i in range(len(dx)):
    for j in range(len(dx[i])):
        if dx[i][j] != 0:
            dx_e = True
            print("%sth atom's %sth dx value initialize not correctly"%(i+1, j+1))
if not dx_e:
    print"dx initialize ok"

for i in range(len(da)):
    for j in range(len(da[i])):
        if da[i][j] != 0:
            da_e = True
            print("%sth atom's %sth da value initialize not correctly"%(i+1, j+1))
if not da_e:
    print"da initialize ok"
