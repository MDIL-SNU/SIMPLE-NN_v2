import sys
import os
sys.path.append('./')
import numpy as np
import torch 
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.features.mpi import DummyMPI
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

rootdir='./test_input/preprocess/'
logfile = open(rootdir+'LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)
 

print('_calculate_scale test')
train_feature_list, train_idx, train_dir  = util_ft._make_full_featurelist(rootdir+'train_list', 'x', inputs['atom_types'])
print('generate train_feature_list done ')
print(f"Road pregenerated featurelist : {rootdir}feature_match")
#Generate save
#torch.save([train_feature_list, train_idx, train_dir], rootdir+'feature_match')

feature_match, train_idx, train_dir = torch.load(f"{rootdir}feature_match")

if np.sum(train_feature_list['Si']-feature_match['Si']) == 0.0:
    print("Same train_feature_list generated")
else:
    raise Exception("Error occoured : other value generated in train_feature_list")

comm = DummyMPI()

scale = preprocessing._calculate_scale(inputs, logfile, train_feature_list, comm)
print("generate scale factor done")
print(f"Road pregenerated scale : {rootdir}scale_match")

#Generate value
#torch.save(scale, rootdir+'scale_match')
scale_match = torch.load(f"{rootdir}scale_match")

if np.sum(scale['Si'] - scale_match['Si']) == 0.0:
    print("Same scale_factor generated")
else:
    raise Exception("Error occoured : other value generated in scale_factor")

print('_calculate_scale OK')
print('')

