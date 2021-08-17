import sys
import os
sys.path.append('./')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#os.system('python libsymf_builder.py')

import torch
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


print('save_to_datafile() test')
data = {'TMP': 2, 'DATA': 5}
data_idx = 3
tag_idx = 3
print('test dictionary data ', data)
print('tag_index ', tag_idx)
tmp_filename = data_generator.save_to_datafile(inputs, data, data_idx, logfile)
print('tmp_filename    :', tmp_filename)

print("load saved data : ./data/data3.pt")
if os.path.exists('./data/data3.pt'):
    saved_data = torch.load("./data/data3.pt")
    assert saved_data['TMP'] == 2 , f"Error occred : not consistant result"
    assert saved_data['DATA'] == 5 , f"Error occred : not consistant result"
else:
    raise Exception("Error occured : data not saved")

print("Remove generated data ./data/data3.pt")
os.system("rm -r ./data/")
print('save_to_pickle() OK')
print('')

