import sys
import os
sys.path.append('./')
import numpy as np
import torch 
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

rootdir='./test_input/'
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)
 

print('_calculate_scale test')
train_feature_list = util_ft._make_full_featurelist('./train_list', 'x', inputs['atom_types'], pickle_format=False)
print('generate train_feature_list done ')
print(f"Road pregenerated featurelist : {rootdir}feature_match")
feature_match = torch.load(f"{rootdir}feature_match")
if np.sum(train_feature_list['Si']-feature_match['Si']) == 0.0:
    print("Same train_feature_list generated")
else:
    raise Exception("Error occoured : other value generated in train_feature_list")

scale = preprocessing._calculate_scale(inputs, logfile, train_feature_list)
print("generate scale factor done")
print(f"Road pregenerated scale : {rootdir}scale_match")

scale_match = torch.load(f"{rootdir}scale_match")
if np.sum(scale['Si'] - scale_match['Si']) == 0.0:
    print("Same scale_factor generated")
else:
    raise Exception("Error occoured : other value generated in scale_factor")

print('_calculate_scale OK')
print('')

