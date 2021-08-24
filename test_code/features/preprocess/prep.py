import sys
import os
sys.path.append('./')



from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.features.mpi import DummyMPI
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

rootdir='./test_input/preprocess/'
logfile = open(rootdir+'LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)
 

print('preprocess() test')
os.system(f'cp {rootdir}total_list ./')
comm = DummyMPI()
preprocessing.preprocess(inputs, logfile, comm)

filelist = [
    'train_list',
    'valid_list',
    ]
matchlist = [
    'train_match',
    'valid_match',
    ]

print("Match generated files")
for it in range(len(filelist)):
    print(f"test match {filelist[it]} with {matchlist[it]}")
    match_info = os.system(f"diff ./{filelist[it]} {rootdir}{matchlist[it]}")
    if match_info == 0:
        print(f"same result  {filelist[it]} with {matchlist[it]}")
    elif match_info == 1:
        raise Exception(f"Error occured : Not same text file  {filelist[it]} with {matchlist[it]}")
    elif match_info == 2:
        raise Exception(f"Error occured : Not same binary file {filelist[it]} with {matchlist[it]}")
    else:
        raise Exception(f"Unexpected error occured : code {match_info}")

print('preprocess() OK')
print('clear generated files in preprocess')
os.system('rm LOG total_list train_list valid_list pca scale_factor')
print('')


