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

print('_get_tag_and_weight() test')
test_text = ['OUTCAR1','OUTCAR2:3','OUTCAR3 : 3']
tag_ans= ['OUTCAR1','OUTCAR2','OUTCAR3']
weight_ans = [1.0,3.0,3.0]
print("temporary file list : " ,test_text) 

for it, text in enumerate(test_text):
    print('Main text   :   ' + text)
    tag , weight = data_generator._get_tag_and_weight(text)
    print('Tag         :  ' + tag)
    assert tag == tag_ans[it] , f"Error occred : not match with data, {tag}"
    print('Weight      :  ', weight)
    assert weight == weight_ans[it] , f"Error occred : not match with data, {weight}"
    print('')

print('_get_tag_and_weight() OK')
print('')

