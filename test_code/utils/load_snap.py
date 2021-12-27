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
from simple_nn.features.mpi import DummyMPI

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
rootdir = './test_input/utils/'
logfile = open(rootdir+'LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)

atom_types = inputs['atom_types']
structure_list = rootdir+'structure_list'




print('load_snapshots() test')
test_file_list = ['test_data/GST/OUTCAR_1', 'test_data/GST/OUTCAR_2', 'test_data/GST/OUTCAR_3']
test_slicing_list = ['::3', '::', '::']



str_match = None
if os.path.exists(rootdir+'str_match'):
    print("load structure data to match")
    str_match = torch.load(rootdir+'str_match')
    print("Loaded data : \n", str_match)
    print("Done \n")

save_dict = dict()

for test_file, test_slicing in zip(test_file_list, test_slicing_list):
    print('Main structure  :   ',rootdir+test_file)
    structures = data_generator.load_structures(inputs, './test_input/'+test_file, test_slicing, logfile, comm=DummyMPI())
    save_dict[test_file] = structures
    print('Structures        :  ',structures)
    #if str_match:
    #    assert structures == str_match[test_file], f"Error occured : not consistant match structure {test_file}"
    print('')

if not os.path.exists(rootdir+'str_match'):
    torch.save(save_dict, rootdir+'str_match')


print('load_structures() OK')
print('')
