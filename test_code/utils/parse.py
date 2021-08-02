import sys
import os
sys.path.append('./')
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#os.system('python libsymf_builder.py')

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating
from simple_nn_v2.utils import data_generator

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
rootdir = './test_input/'
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)

atom_types = inputs['atom_types']
structure_list = rootdir+'structure_list'


atype = 'SiO2 type'
weight = '10.0'
list_dir = './test_data/SiO2/OUTCAR_comp' 
sliced   = ':30:10'

if not os.path.exists(structure_list):
    print("structure_list not exist. create structure_list")
    os.system("touch "+structure_list)
    os.system(f"echo '[{atype} : {weight}]\n' >> "+structure_list)
    os.system(f"echo '{list_dir} {sliced}' >> "+structure_list)





print('parse_structure_list() test')
structure_tags, structure_weights, structure_file_list, structure_slicing_list, structure_tag_idx = data_generator.parse_structure_list(logfile, structure_list)
print('structure_tags   :', structure_tags)
assert structure_tags[0] == 'None' , f"Error occured : not match tags  {structure_tags[0]}"
assert structure_tags[1] == atype , f"Error occured : not match tags  {structure_tags[1]}"
print("structure_tags compare OK")


print('structure_weights   :', structure_weights)
assert structure_weights[1] == float(weight) , f"Error occured : not consistant structure_weights"
print("structure_weights compare OK")

print('structure_file_list   :', structure_file_list)
assert structure_file_list[0] == list_dir , f"Error occured : not consistant structure_file_list"
print("structure_file_list compare OK")


print('structure_slicing_list   :', structure_slicing_list)
assert structure_slicing_list[0] == sliced , f"Error occured : not consistant structure_slicing_list"
print("structure_file_list compare OK")

print('structure_tag_idx   :', structure_tag_idx)
assert structure_tag_idx[0] == 1 , f"Error occured : not consistant structure_tag_idx"
print("structure_file_list compare OK")

print('parse_structure_list() OK')

