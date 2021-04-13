## Before the training, pickle data are changed into .tfrecord format. (Later .pt format)
## This class conatin Scale Factor, PCA, GDF, weight calculations
##        
##
##          :param boolean calc_scale: flag to calculate the scale_factor
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
def preprocess(inputs, logfile, calc_scale=True, get_atomic_weights=None, **kwargs):
        """
        Before the training, pickle data are changed into .tfrecord format.
        scale_factor, pca, gdf are also calculated here.

        :param boolean calc_scale: flag to calculate the scale_factor
        :param boolean use_force: flag to put 'force data' into input
        :param boolean use_stress: flag to put 'stress data' into the input
        :param boolean get_atomic_weights: flag to return atomic_weight
        """ 
        #Extract save directory information from save_dir
        structure_list = './str_list' 
        pickle_list = './pickle_list'
        train_data_list = './train_list'
        valid_data_list = './valid_list'


        is_ptdata = not inputs['symmetry_function']['save_to_pickle']

        #Split Test, Valid data of str_list
        tmp_pickle_train, tmp_pickle_valid =_split_data(inputs)        
        # generate full symmetry function vector
        feature_list_train, idx_list_train = \
            _make_full_featurelist(tmp_pickle_train, 'x', inputs['atom_types'], is_ptdata=is_ptdata)
        feature_list_valid, idx_list_valid = \
            _make_full_featurelist(tmp_pickle_valid, 'x', inputs['atom_types'], is_ptdata=is_ptdata)
        # calculate scale
        scale = _calculate_scale(inputs, feature_list_train ,logfile, calc_scale=calc_scale)
        # Fit PCA.
        _generate_pca(inputs, feature_list_train, scale)
        # calculate gdf
        aw_tag, atomic_weights_train, atomic_weights_valid = _calculate_gdf(inputs, \
        feature_list_train, idx_list_train, feature_list_valid, idx_list_valid, logfile, get_atomic_weights,**kwargs)
        #Make tfrecord file from training
        _dump_all(inputs, tmp_pickle_train, atomic_weights_train, train_data_list,\
        tmp_pickle_valid, atomic_weights_valid, valid_data_list, aw_tag, logfile,is_tfrecord=True)
        #END of preprocess


    #This function split train, valid list 
def _split_data(inputs, pickle_list='./pickle_list'):
    tmp_pickle_train = './pickle_train_list'
    tmp_pickle_valid = './pickle_valid_list'
            
    if not inputs['neural_network']['continue']:
       tmp_pickle_train_open = open(tmp_pickle_train, 'w')
       tmp_pickle_valid_open = open(tmp_pickle_valid, 'w')
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

    return tmp_pickle_train, tmp_pickle_valid

    #This function calculate scale factor as pickle
def _calculate_scale(inputs, feature_list_train, logfile, calc_scale=True):
    scale = None
    params_set = dict()
    for item in inputs['atom_types']:
        params_set[item] = dict()
        params_set[item]['i'], params_set[item]['d'] = _read_params(inputs['symmetry_function']['params'][item])
    if calc_scale:
        scale = _generate_scale_file(feature_list_train, inputs['atom_types'], 
                                     scale_type=inputs['symmetry_function']['scale_type'],
                                     scale_scale=inputs['symmetry_function']['scale_scale'],
                                     scale_rho=inputs['symmetry_function']['scale_rho'],
                                     params=params_set,
                                     log=logfile,
                                     comm=DummyMPI())
    else:
        scale = pickle_load('./scale_factor')
    return scale 

    #This function to generate PCA data
def _generate_pca(inputs, feature_list_train, scale):
    if inputs['neural_network']['pca']:
        pca = {}
        for item in inputs['atom_types']:
            pca_temp = PCA()
            pca_temp.fit((feature_list_train[item] - scale[item][0:1,:]) / scale[item][1:2,:])
            min_level = inputs['neural_network']['pca_min_whiten_level']
            # PCA transformation = x * pca[0] - pca[2] (divide by pca[1] if whiten)
            pca[item] = [pca_temp.components_.T,
                         np.sqrt(pca_temp.explained_variance_ + min_level),
                         np.dot(pca_temp.mean_, pca_temp.components_.T)]
        with open("./pca", "wb") as fil:
            pickle.dump(pca, fil, protocol=2)

    #This function for calculate GDF        
def _calculate_gdf(inputs, feature_list_train, idx_list_train, featrue_list_valid, idx_list_valid,logfile ,get_atomic_weights=None, **kwargs):
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

#This function train data and create .tfrecord file is_tfrecord input is for debugging
def _dump_all(inputs,  tmp_pickle_train, atomic_weights_train, train_data_list,\
    tmp_pickle_valid, atomic_weights_valid, valid_data_list, aw_tag, logfile, is_tfrecord=True):
    # Start of training, validation

    #Extracty data_per_tfrecord from inputs & save_directory
    data_per_file = inputs['symmetry_function']['data_per_tfrecord']
    save_dir = inputs['symmetry_function']['save_directory']
    
    #Create save directory
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    tmp_pickle_train_list = _make_data_list(tmp_pickle_train)
    #np.random.shuffle(tmp_pickle_train_list)
    num_tmp_pickle_train = len(tmp_pickle_train_list)
    num_tfrecord_train = int(num_tmp_pickle_train / data_per_file)
    train_list = open(train_data_list, 'w')

    random_idx = np.arange(num_tmp_pickle_train)        
    if inputs['symmetry_function']['shuffle']:
        np.random.shuffle(random_idx)
    
    for i,item in enumerate(random_idx):
        ptem = tmp_pickle_train_list[item]
        if is_tfrecord: #Save to .tfrecord file
            if i == 0:
                record_name = save_dir+'/training_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/data_per_file), num_tfrecord_train)
                writer = tf.python_io.TFRecordWriter(record_name)
            elif (i % data_per_file) == 0:
                writer.close()
                logfile.write('{} file saved in {}\n'.format(data_per_file, record_name))
                train_list.write(record_name + '\n')
                record_name = save_dir+'/training_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/data_per_file), num_tfrecord_train)
                writer = tf.python_io.TFRecordWriter(record_name)
        else: #Save to pytorch file
            #TODO: make for torch version to make trrecord file
            pass
        
        #Check data was saved to .pt file format or .pickle file format
        if inputs['symmetry_function']['save_to_pickle']:
            tmp_res = pickle_load(ptem)
        else:
            tmp_res = torch.load(ptem)
            
        tmp_res['pickle_name'] = ptem
        #Check atomic_weights_train exists
        if atomic_weights_train is not None:
            tmp_aw = dict()
            for jtem in inputs['atom_types']:
                tmp_idx = (atomic_weights_train[jtem][:,1] == item)
                tmp_aw[jtem] = atomic_weights_train[jtem][tmp_idx,0]
            #tmp_aw = np.concatenate(tmp_aw)
            tmp_res['atomic_weights'] = tmp_aw
        
        #write tfrecord format
        if is_tfrecord:
            _write_tfrecords(inputs, tmp_res, writer, atomic_weights=aw_tag)
        else:
            #TODO: make torch version of  _write_tfrecords 
            pass

        if not inputs['symmetry_function']['remain_pickle']:
            os.remove(ptem)

    writer.close()
    logfile.write('{} file saved in {}\n'.format((i%data_per_file)+1, record_name))
    train_list.write(record_name + '\n')
    train_list.close()
    
    
    #Start validation part
    if inputs['symmetry_function']['valid_rate'] != 0.0:
        # valid
        tmp_pickle_valid_list = _make_data_list(tmp_pickle_valid)
        num_tmp_pickle_valid = len(tmp_pickle_valid_list)
        num_tfrecord_valid = int(num_tmp_pickle_valid / data_per_file)
        valid_list = open(valid_data_list, 'w')

        random_idx = np.arange(num_tmp_pickle_valid)        
        if inputs['symmetry_function']['shuffle']:
            np.random.shuffle(random_idx)

        for i,item in enumerate(random_idx):
            ptem = tmp_pickle_valid_list[item]
            if is_tfrecord: #tfrecord version
                if i == 0:
                    record_name = save_dir+'/valid_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/data_per_file), num_tfrecord_valid)
                    writer = tf.python_io.TFRecordWriter(record_name)
                elif (i % data_per_file) == 0:
                    writer.close()
                    logfile.write('{} file saved in {}\n'.format(data_per_file, record_name))
                    valid_list.write(record_name + '\n')
                    record_name = save_dir+'/valid_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/data_per_file), num_tfrecord_valid)
                    writer = tf.python_io.TFRecordWriter(record_name)
            else: #pytorch version
                #TODO: make torch version to save tfrecord
                pass

            if inputs['symmetry_function']['save_to_pickle']:
                tmp_res = pickle_load(ptem)
            else:
                tmp_res = torch.load(ptem)
            tmp_res['pickle_name'] = ptem
            
            #Check atomic_weights_valid 
            if atomic_weights_valid == 'ones':
                tmp_aw = dict()
                for jtem in inputs['atom_types']:
                    tmp_aw[jtem] = np.ones([tmp_res['N'][jtem]]).astype(np.float64)
                tmp_res['atomic_weights'] = tmp_aw
            elif atomic_weights_valid is not None:
                tmp_aw = dict()
                for jtem in inputs['atom_types']:
                    tmp_idx = (atomic_weights_valid[jtem][:,1] == item)
                    tmp_aw[jtem] = atomic_weights_valid[jtem][tmp_idx,0]
                #tmp_aw = np.concatenate(tmp_aw)
                tmp_res['atomic_weights'] = tmp_aw

            if is_tfrecord:
                _write_tfrecords(inputs, tmp_res, writer, atomic_weights=aw_tag)
            else:
                #TODO: Make torch version of _write_tfrecords function
                pass

            if not inputs['symmetry_function']['remain_pickle']:
                os.remove(ptem)

        writer.close()
        logfile.write('{} file saved in {}\n'.format((i%data_per_file)+1, record_name))
        valid_list.write(record_name + '\n')
        valid_list.close()


## Temporary code for tfrecords
def _write_tfrecords(inputs, res, writer, atomic_weights=False):
    # TODO: after stabilize overall tfrecord related part,
    # this part will replace the part of original 'res' dict
     
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _gen_1dsparse(arr):
        non_zero = (arr != 0)
        return np.arange(arr.shape[0])[non_zero].astype(np.int32), arr[non_zero], np.array(arr.shape).astype(np.int32)
    
    feature = {
        'E':_bytes_feature(np.array([res['E']]).tobytes()),
        'tot_num':_bytes_feature(res['tot_num'].astype(np.float64).tobytes()),
        'partition':_bytes_feature(res['partition'].tobytes()),
        'struct_type':_bytes_feature(six.b(res['struct_type'])),
        'struct_weight':_bytes_feature(np.array([res['struct_weight']]).tobytes()),
        'pickle_name':_bytes_feature(six.b(res['pickle_name'])),
        'atom_idx':_bytes_feature(res['atom_idx'].tobytes())
    }

    try:
        feature['F'] = _bytes_feature(res['F'].tobytes())
    except:
        pass

    try:
        feature['S'] = _bytes_feature(res['S'].tobytes())
    except:
        pass

    for item in inputs['atom_types']:
        feature['x_'+item] = _bytes_feature(res['x'][item].tobytes())
        feature['N_'+item] = _bytes_feature(res['N'][item].tobytes())
        feature['params_'+item] = _bytes_feature(res['params'][item].tobytes())

        dx_indices, dx_values, dx_dense_shape = _gen_1dsparse(res['dx'][item].reshape([-1]))
    
        feature['dx_indices_'+item] = _bytes_feature(dx_indices.tobytes())
        feature['dx_values_'+item] = _bytes_feature(dx_values.tobytes())
        feature['dx_dense_shape_'+item] = _bytes_feature(dx_dense_shape.tobytes())

        feature['da_'+item] = _bytes_feature(res['da'][item].tobytes())

        feature['partition_'+item] = _bytes_feature(res['partition_'+item].tobytes())

        if atomic_weights:
            feature['atomic_weights_'+item] = _bytes_feature(res['atomic_weights'][item].tobytes())

        if inputs['symmetry_function']['add_NNP_ref']:
            feature['NNP_E_'+item] = _bytes_feature(res['NNP_E'][item].tobytes())

    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )
    
    writer.write(example.SerializeToString())


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
