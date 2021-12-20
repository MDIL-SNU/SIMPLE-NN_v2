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
from simple_nn_v2.features.mpi import DummyMPI, MPI4PY


import numpy as np
import torch 


rootdir = './test_input/gdf'
def test():
    comm = DummyMPI()
    inputs = dict()
    logfile = open('LOG','w')
    inputs['atom_types'] = ['Si']
    inputs['preprocessing'] = { 'calc_atomic_weights':{'type':'gdf','params':{'sigma':0.02}}, 'valid_list':None, 'weight_modifier':{'type':None},
     'calc_scale':True, 'scale_type':'minmax', 'scale_width':1.0,'scale_rho':None, 'valid_rate':0.0}

    #Temporary symmetry function
    train_feature_list = {'Si':np.array([
    [0.8],
    [0.7],
    [0.1],
    [0.11],
    [0.12],
    [0.13],
    [0.14],
    [0.15],
    ])}
    train_idx_list = {'Si':np.array([0,0,1,1,2,2,3,3])}
    train_dir_list = [f'{rootdir}/data{i+1}.pt' for i in range(int(len(train_idx_list['Si'])/2))]
    tmp_idx = 0
    for fil in train_dir_list:
        tmp = {'atom_idx':np.array([1,1]),'tot_num':2, 'N':{'Si':2}}
        tmp['atom_types'] = ['Si']
        tmp['x'] = {'Si':train_feature_list['Si'][tmp_idx:tmp_idx+2,:]}
        print("Saved at ",fil)
        torch.save(tmp, fil)
        tmp_idx += 2
    print('total feature list (x)')
    print(train_feature_list)



    #Get scale
    scale_minmax = _calculate_scale(inputs, logfile, train_feature_list, comm)
    print('Scale : minmax')
    print(scale_minmax)
    inputs['preprocessing']['scale_type'] = 'meanstd'
    scale_meanstd = _calculate_scale(inputs, logfile, train_feature_list, comm)
    print('Scale : minstd')
    print(scale_meanstd)

    _calculate_gdf(inputs, logfile, train_feature_list, train_idx_list ,train_dir_list, scale_minmax ,comm)

    for fil in train_dir_list:
        if os.path.exists(fil):
            print('_________________________________________________')
            print(fil)
            print('Atom index')
            print(torch.load(fil)['atom_idx'])
            print('Symmetry function')
            print(torch.load(fil)['x'])
            print('Gaussian density function ')
            print(torch.load(fil)['atomic_weights'])
            print('_________________________________________________')

    print('\n\n')
    
    #Calculate GDF factor
    def get_gdf(distance_list , sigma=0.02, dim = 1):
        out = list(map(lambda distance:np.exp(-(distance**2)/(2*sigma**2*dim)), distance_list))
        out = np.sum(out) / len(distance_list)
        return 1/out

    test_feature_list = np.array([0.8,0.7,0.1,0.11,0.12,0.13,0.14,0.15]) 

    scaled_feature_list   = test_feature_list - scale_minmax['Si'][0:1,:]
    scaled_feature_list    /= scale_minmax['Si'][1:2,:]
    #print(test_feature_list) 
    #print(scaled_feature_list[0]) 
    #print(get_gdf(test_feature_list - test_feature_list[0]))
    gdf = list()
    for idx in range(len(test_feature_list)):
        #print(scaled_feature_list[0] - scaled_feature_list[0][idx])
        print(f'IDX {idx} GDF : ',get_gdf(scaled_feature_list[0] - scaled_feature_list[0][idx]))
        gdf.append(get_gdf(scaled_feature_list[0] - scaled_feature_list[0][idx]))


    reduced_gdf = np.array(gdf) # / np.mean(gdf)

    print('_________________________________________________')
    print('Assertion Check')
    for idx, fil in enumerate(train_dir_list):
        if os.path.exists(fil):
            tmp_val_even = np.abs(torch.load(fil)['atomic_weights'][0] - reduced_gdf[2*idx])
            tmp_val_odd = np.abs(torch.load(fil)['atomic_weights'][1] - reduced_gdf[2*idx+1])
            assert (tmp_val_even < 1E-2), f"Error : Wrong GDF value diff {tmp_val_even} "
            assert (tmp_val_odd < 1E-2), f"Error : Wrong GDF value diff {tmp_val_odd} "
            print(fil + ' passed ')
    print('_________________________________________________')

    #Check GDF sigma = Auto Part
    inputs['preprocessing']['calc_atomic_weights']['params']['sigma'] = 'Auto'
    _calculate_gdf(inputs, logfile, train_feature_list, train_idx_list ,train_dir_list, scale_minmax, comm)

if __name__ == '__main__':
    test()




