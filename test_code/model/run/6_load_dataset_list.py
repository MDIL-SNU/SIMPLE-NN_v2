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
print(train_dataset_list.filelist)
print(train_dataset_list[0]['E'])
print(valid_dataset_list.filelist)
print(valid_dataset_list[0]['E'])
