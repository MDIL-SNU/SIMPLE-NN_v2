import sys
import os
sys.path.append('./')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#os.system('python libsymf_builder.py')

import torch
from simple_nn import simple_nn
from simple_nn.init_inputs import initialize_inputs
from simple_nn.features.symmetry_function import generating
from simple_nn.features import data_generator

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
rootdir = './test_input/utils/'
logfile = open(rootdir+'LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)

atom_types = inputs['atom_types']
structure_list = rootdir+'structure_list'


print('save_to_datafile() test')
data = {'TMP': 2, 'DATA': 5}
data_idx = 3
tag_idx = 3
print('test dictionary data ', data)
print('tag_index ', tag_idx)
print(f'{rootdir}data dirctory maden')
if not os.path.exists(f'{rootdir}data'):
    os.mkdir(f'{rootdir}data')

tmp_filename = data_generator.save_to_datafile(inputs, data, data_idx, logfile)
print('tmp_filename    :', tmp_filename)


print(f"load saved data : {rootdir}data/data3.pt")
if os.path.exists(f'{rootdir}data/data3.pt'):
    saved_data = torch.load(f"{rootdir}data/data3.pt")
    assert saved_data['TMP'] == 2 , f"Error occred : not consistant result"
    assert saved_data['DATA'] == 5 , f"Error occred : not consistant result"
    print('{rootdir}data/data3.pt is saved well')
else:
    raise Exception("Error occured : data not saved")

print('save_to_pickle() OK')
print('')

