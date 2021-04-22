## Before the training, pickle data are changed into .tfrecord format. (Later .pt format)
## This class conatin Scale Factor, PCA, GDF, weight calculations
##        
##
##          :param boolean use_force: flag to put 'force data' into input
##          :param boolean use_stress: flag to put 'stress data' into the input
##          :param boolean get_atomic_weights: flag to return atomic_weight 
##
##

import os, sys
import tensorflow as tf
import torch

import numpy as np
import six
from six.moves import cPickle as pickle

from ..utils import _generate_scale_file, _make_full_featurelist, _make_data_list, \
                    _make_str_data_list, pickle_load
from ..utils import graph as grp
from ..utils.mpiclass import DummyMPI

from sklearn.decomposition import PCA

#Do all preprocess 
def preprocess(inputs, logfile, get_atomic_weights=None, **kwargs):
        """
        Before the training, pickle data are changed into .tfrecord format.
        scale_factor, pca, gdf are also calculated here.

        :param boolean use_force: flag to put 'force data' into input
        :param boolean use_stress: flag to put 'stress data' into the input
        :param boolean get_atomic_weights: flag to return atomic_weight
        """ 
        #Extract save directory information from save_dir
        pickle_list = './pickle_list'

        is_ptdata = not inputs['symmetry_function']['save_to_pickle']

        #Split Test, Valid data of str_list
        _split_data(inputs, pickle_list)        
        # generate full symmetry function vector
        feature_list_train, idx_list_train = \
            _make_full_featurelist(inputs['symmetry_function']['train_list'], 'x', inputs['atom_types'], is_ptdata=is_ptdata)
        feature_list_valid, idx_list_valid = \
            _make_full_featurelist(inputs['symmetry_function']['valid_list'], 'x', inputs['atom_types'], is_ptdata=is_ptdata)
        # calculate scale
        scale = _calculate_scale(inputs, logfile, feature_list_train)
        # Fit PCA.
        _generate_pca(inputs, logfile, feature_list_train, scale)
        # calculate gdf
        aw_tag, atomic_weights_train, atomic_weights_valid = _calculate_gdf(inputs, logfile,\
        feature_list_train, idx_list_train, feature_list_valid, idx_list_valid, get_atomic_weights,**kwargs)
        #END of preprocess


    #This function split train, valid list 
def _split_data(inputs, pickle_list='./pickle_list'):
    if not inputs['neural_network']['continue']:
       tmp_pickle_train_open = open(inputs['symmetry_function']['train_list'], 'w')
       tmp_pickle_valid_open = open(inputs['symmetry_function']['valid_list'], 'w')
       for file_list in _make_str_data_list(pickle_list):
           if inputs['symmetry_function']['shuffle']:
               np.random.shuffle(file_list)
           num_pickle = len(file_list)
           num_valid = int(num_pickle * inputs['symmetry_function']['valid_rate'])

           for i,item in enumerate(file_list):
               if i < num_valid:
                   tmp_pickle_valid_open.write(item + '\n')
               else:
                   tmp_pickle_train_open.write(item + '\n')
       tmp_pickle_train_open.close()
       tmp_pickle_valid_open.close()


    #This function calculate scale factor as pickle
def _calculate_scale(inputs, logfile, feature_list_train):
    scale = None
    params_set = dict()
    feature_dict = dict() # Added
    for item in inputs['atom_types']:
        params_set[item] = dict()
        params_set[item]['i'], params_set[item]['d'] = _read_params(inputs['symmetry_function']['params'][item])
        feature_dict[item] = list()
    if inputs['symmetry_function']['calc_scale']:
        scale = _generate_scale_file(feature_list_train, inputs['atom_types'], 
                                     scale_type=inputs['symmetry_function']['scale_type'],
                                     scale_scale=inputs['symmetry_function']['scale_scale'],
                                     scale_rho=inputs['symmetry_function']['scale_rho'],
                                     params=params_set,
                                     log=logfile,
                                     comm=DummyMPI())
    else:
        scale = torch.load('./scale_factor')
    return scale 

    #This function to generate PCA data
def _generate_pca(inputs, logfile ,feature_list_train, scale):
    if inputs['neural_network']['pca']:
        pca = dict()
        scale_process = None
        for item in inputs['atom_types']:
            pca_temp = PCA()
            scale_process = (feature_list_train[item] - scale[item][0].reshape(1,-1) )  / scale[item][1].reshape(1,-1)
            pca_temp.fit(scale_process)
            min_level = inputs['neural_network']['pca_min_whiten_level'] if inputs['neural_network']['pca_min_whiten_level'] else 0.0
            # PCA transformation = x * pca[0] - pca[2] (divide by pca[1] if whiten)
            pca[item] = [pca_temp.components_.T,
                         np.sqrt(pca_temp.explained_variance_ + min_level),
                         np.dot(pca_temp.mean_, pca_temp.components_.T)]
        logfile.write('PCA complete\n')
        torch.save(pca, './pca')

    #This function for calculate GDF        
def _calculate_gdf(inputs, logfile, feature_list_train, idx_list_train, featrue_list_valid, idx_list_valid, get_atomic_weights=None, **kwargs):
    #Define outputs
    atomic_weights_train = atomic_weights_valid = None
    
    #Check get_atomic_weigts exist
    if callable(get_atomic_weights):
        # FIXME: use mpi
        local_target_list = dict()
        local_idx_list = dict()
        #feature_list_train part
        for item in inputs['atom_types']:

            local_target_list[item] = feature_list_train[item]
            local_idx_list[item] = idx_list_train[item]

        atomic_weights_train, dict_sigma, dict_c = get_atomic_weights(feature_list_train, scale, inputs['atom_types'], local_idx_list, 
                                                                      target_list=local_target_list, filename='atomic_weights', comm=DummyMPI(), **kwargs)
        kwargs.pop('sigma')

        logfile.write('Selected(or generated) sigma and c\n')
        for item in inputs['atom_types']:
            logfile.write('{:3}: sigma = {:4.3f}, c = {:4.3f}\n'.format(item, dict_sigma[item], dict_c[item]))

        local_target_list = dict()
        local_idx_list = dict()
        #feature_list_valid part
        for item in inputs['atom_types']:

            local_target_list[item] = feature_list_valid[item]
            local_idx_list[item] = idx_list_valid[item]

        atomic_weights_valid,          _,        _ = get_atomic_weights(feature_list_train, scale, inputs['atom_types'], local_idx_list, 
                                                                      target_list=local_target_list, sigma=dict_sigma, comm=DummyMPI(), **kwargs)
    #Check get_atomic_weights is six.string_types
    elif isinstance(get_atomic_weights, six.string_types):
        atomic_weights_train = pickle_load(get_atomic_weights)
        atomic_weights_valid = 'ones'

    if atomic_weights_train is None:
        aw_tag = False
    else:
        aw_tag = True
        #grp.plot_gdfinv_density(atomic_weights_train, inputs['atom_types'])
        # Plot histogram only if atomic weights just have been calculated.
        if callable(get_atomic_weights):
            grp.plot_gdfinv_density(atomic_weights_train, inputs['atom_types'], auto_c=dict_c)
    return aw_tag, atomic_weights_train, atomic_weights_valid


    #This function for read parameters of symmetry functions
def _read_params(filename):
    params_i = list()
    params_d = list()
    with open(filename, 'r') as fil:
        for line in fil:
            tmp = line.split()
            params_i += [list(map(int, tmp[:3]))]
            params_d += [list(map(float, tmp[3:]))]

    params_i = np.asarray(params_i, dtype=np.intc, order='C')
    params_d = np.asarray(params_d, dtype=np.float64, order='C')

    return params_i, params_d
