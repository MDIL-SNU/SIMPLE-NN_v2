import sys
import os
sys.path.append('./')
sys.path.append('../../../')



from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

rootdir='./test_input/'
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)
 
print('_split_train_list_and_valid_list test : '+rootdir+'tota_list')
print('total_list : ')
os.system(f'cat {rootdir}total_list')
out_list = preprocessing._split_train_list_and_valid_list(inputs, data_list=rootdir+'total_list')


diff_train = os.system(f'diff ./train_list {rootdir}train_match')
if diff_train == 0:
    print("Same list generated with pregenerated train_list")
elif diff_train == 1:
    print("Error occured : Not same list generated with pregenerated train_list")
    raise Exception('Error occured : train_list in _split_train_list_and_valid_list')
else:
    raise Exception("Error occured : unexpected error occured")

diff_valid = os.system(f'diff ./valid_list {rootdir}valid_match')
if diff_valid == 0:
    print("Same list generated with pregenerated valid_list")
elif diff_valid == 1:
    print("Error occured : Not same list generated with pregenerated valid_list")
    raise Exception('Error occured : train_list in _split_train_list_and_valid_list')
else:
    raise Exception("Error occured : unexpected error occured")

print('')



