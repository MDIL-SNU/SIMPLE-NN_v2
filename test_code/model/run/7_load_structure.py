import sys
sys.path.insert(0, '../../../')
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.models import run

import torch

logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('./input.yaml', logfile)
atom_types = inputs['atom_types']

torch.set_default_dtype(torch.float64)
train_dataset_list, valid_dataset_list = run._load_dataset_list(inputs, logfile)
scale_factor = torch.load('scale_factor')
pca = torch.load('pca')

train_struct_dict, valid_struct_dict = run._load_structure(inputs, logfile, scale_factor, pca)

for item in train_struct_dict:
    print(train_struct_dict[item])

