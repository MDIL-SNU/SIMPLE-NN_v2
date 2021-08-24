import sys
import os
sys.path.append('./')
import torch

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2.features.preprocessing import _calculate_gdf
from simple_nn_v2.features.preprocessing import _calculate_scale
from simple_nn_v2.features.symmetry_function.mpi import DummyMPI, MPI4PY


import numpy as np
import torch 
rootdir='./test_input/gdf/'
def test():
    comm = DummyMPI()
    inputs = dict()
    logfile = open(rootdir+'LOG','w')
    inputs['atom_types'] = ['Si']
    inputs['preprocessing'] = { 'atomic_weights':{'type':'gdf'}, 'valid_list':None, 'weight_modifier':{'type':None},
     'calc_scale':True, 'scale_type':'minmax', 'scale_scale':1.0,'scale_rho':None, 'valid_rate':0.0}

    #Temporary symmetry function
    train_feature_list = {'Si':np.array([
    [0.8,0,0,0],
    [0.11,0,0,0],
    [0.12,0,0,0],
    [0.13,0,0,0],
    [0.14,0,0,0],
    [0.15,0,0,0],
    ])}
    train_idx_list = {'Si':np.array([0,0,1,1,2,2])}
    train_dir_list = [rootdir+i for i in ['data1.pt','data2.pt','data3.pt']]
    tmp_idx = 0
    for fil in train_dir_list:
        tmp = {'atom_idx':np.array([1,1]),'tot_num':2, 'N':{'Si':2}}
        tmp['atom_types'] = ['Si']
        tmp['x'] = {'Si':train_feature_list['Si'][tmp_idx:tmp_idx+2,:]}
        torch.save(tmp, fil)
        tmp_idx += 2
    print('total feature list (x)')
    print(train_feature_list)



    #Get scale
    scale = _calculate_scale(inputs, logfile, train_feature_list, comm)
    print('Scale')
    print(scale)

    _calculate_gdf(inputs, logfile, train_feature_list, train_idx_list ,train_dir_list, scale, comm )

    for fil in train_dir_list:
        if os.path.exists(fil):
            print('_________________________________________________')
            print('atom_idx')
            print(torch.load(fil)['atom_idx'])
            print('x')
            print(torch.load(fil)['x'])
            print('gdf')
            print(torch.load(fil)['gdf'])
            print('_________________________________________________')



if __name__ == '__main__':
    test()




