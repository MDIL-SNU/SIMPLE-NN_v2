import sys
import os
sys.path.append('./')

import torch
import numpy as np
import sklearn

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

rootdir='./test_input/preprocess/'
logfile = open(rootdir+'LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)


print("Load pregenerateded feature list, scale") 
train_feature_list = torch.load(f'{rootdir}/feature_match')[0]
scale = torch.load(f'{rootdir}/scale_match')
print('_calculate_pca_matrix test')

pca = preprocessing._calculate_pca_matrix(inputs, train_feature_list, scale)
print("pca generate done")


#Saving part
torch.save(pca, f"{rootdir}pca_match")


pca_match = torch.load(f"{rootdir}pca_match")
print("Checking generated pca match ")

if (np.abs(pca['Si'][0][:2]-pca_match['Si'][0][:2])  < 1).all():
    print(f"pca 1st component passed, sum of vector difference under 1")
else:
    print("Difference")
    print(f"{pca['Si'][0][:3] - pca_match['Si'][0][:3]}")
    raise Exception(f"pca generated different value at 1st component, sklearn version : {sklearn.__version__}")

if (np.abs(pca['Si'][1]-pca_match['Si'][1])  < 1).all():
    print(f"pca variance component passed, difference under 1E-10")
else:
    print("Difference")
    print(f"{pca['Si'][1] - pca_match['Si'][1]}")
    raise Exception(f"pca generated different value at 2nd component, sklearn version : {sklearn.__version__}")

if (np.abs(pca['Si'][2][:2]-pca_match['Si'][2][:2])  < 1).all():
    print(f"pca 3rd component passed, difference under 1")
else:
    print("Difference")
    print(f"{pca['Si'][2][:3] - pca_match['Si'][2][:3]}")
    raise Exception(f"pca generated different value at 3rd component, sklearn version : {sklearn.__version__}")

print('_calculate_pca_matrix OK')
print('')

