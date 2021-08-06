import os, sys
import six
import numpy as np
import time
import torch
import functools

from simple_nn_v2.utils import features as util_feature
from simple_nn_v2.utils import scale as util_scale

from simple_nn_v2.features.symmetry_function.mpi import DummyMPI, MPI4PY

from sklearn.decomposition import PCA
 

def preprocess(inputs, logfile, comm):
    """
    1. Split train/valid data names and save in "./pickle_train_list", "./pickle_valid_list" files
    2. Calculate scale factor with symmetry function values of train data set and save as "scale_factor" data (.pt format)
    3. Make PCA matrix with using scikitlearn modules

    Args:
        inputs(dict): full parts in input.yaml
        logfile(file obj): logfile object
    """ 


    data_list = inputs['preprocessing']['data_list']

    _split_train_list_and_valid_list(inputs, data_list)

    # Extract specific feature values('x') from generated data files
    # feature_list.shape(): [(sum of atoms in each data file), (feature length)]
    train_feature_list = util_feature._make_full_featurelist(inputs['preprocessing']['train_list'], 'x', inputs['atom_types'])

    train_feature_list = comm.bcast(train_feature_list)

    # scale[atom_type][0]: (mid_range or mean) of each features
    # scale[atom_type][1]: (width or standard_deviation) of each features
    scale = _calculate_scale(inputs, logfile, train_feature_list, comm)
    torch.save(scale, 'scale_factor')
    logfile.flush()

    # pca[atom_type][0]: principle axis matrix
    # pca[atom_type][1]: variance in each axis
    # pca[atom_type][2]: 
    if inputs['preprocessing']['calc_pca'] and scale:   # boolean :
        pca = _calculate_pca_matrix(inputs, train_feature_list, scale)
        torch.save(pca, 'pca')
        logfile.flush()


    # calculate gdf
    #_calculate_gdf(inputs, logfile, train_feature_list, comm)

# Split train/valid data names that saved in data_list
def _split_train_list_and_valid_list(inputs, data_list='./total_list'):
   train_list_file = open(inputs['preprocessing']['train_list'], 'w')
   valid_list_file = open(inputs['preprocessing']['valid_list'], 'w')

   for file_list in util_feature._make_str_data_list(data_list):
       if inputs['preprocessing']['shuffle'] is True:
           np.random.shuffle(file_list)
       num_pickle = len(file_list)
       num_valid = int(num_pickle * inputs['preprocessing']['valid_rate'])

       for i,elem in enumerate(file_list):
           if i < num_valid:
               valid_list_file.write(elem + '\n')
           else:
               train_list_file.write(elem + '\n')
               
   train_list_file.close()
   valid_list_file.close()

# Calculate scale factor and save as "scale_factor" data (.pt format)
def _calculate_scale(inputs, logfile, feature_list, comm):
    atom_types = inputs['atom_types']
    scale = None
    
    if inputs['preprocessing']['calc_scale'] is True:
        scale = dict()
        scale_type = inputs['preprocessing']['scale_type']
        scale_scale = inputs['preprocessing']['scale_scale']
        calculate_scale_factor = util_scale.get_scale_function(scale_type=scale_type)

        for elem in atom_types:
            inp_size = feature_list[elem].shape[1]
            scale[elem] = np.zeros([2, inp_size])

            # if no feature list, scaling to 1
            if len(feature_list[elem]) <= 0:  
                scale[elem][1,:] = 1.
            else:
                scale[elem][0], scale[elem][1] = calculate_scale_factor(feature_list, elem, scale_scale)
                scale[elem][1, scale[elem][1,:] < 1e-15] = 1.

                is_scaled = np.array([True] * inp_size)
                is_scaled[scale[elem][1,:] < 1e-15] = False

                if logfile is not None and comm.rank == 0:
                    logfile.write("{:-^70}\n".format(" Scaling information for {:} ".format(elem)))
                    logfile.write("(scaled_value = (value - mean) * scale)\n")
                    logfile.write("Index   Mean         Scale        Min(after)   Max(after)   Std(after)\n")
                    scaled = (feature_list[elem] - scale[elem][0,:]) / scale[elem][1,:]
                    scaled_min = np.min(scaled, axis=0)
                    scaled_max = np.max(scaled, axis=0)
                    scaled_std = np.std(scaled, axis=0)
                    for i in range(scale[elem].shape[1]):
                        scale_str = "{:11.4e}".format(1/scale[elem][1,i]) if is_scaled[i] else "Not_scaled"
                        logfile.write("{0:<5}  {1:>11.4e}  {2:>11}  {3:>11.4e}  {4:>11.4e}  {5:>11.4e}\n".format(
                            i, scale[elem][0,i], scale_str, scaled_min[i], scaled_max[i], scaled_std[i]))

        if logfile is not None and comm.rank == 0:
            logfile.write("{:-^70}\n".format(""))
    elif os.path.exists(inputs['preprocessing']['calc_scale']):
        scale = torch.load(inputs['preprocessing']['calc_scale'])

    return scale

# Make PCA matrix with using scikitlearn modules
def _calculate_pca_matrix(inputs, feature_list, scale):
    for elem in inputs['atom_types']:
        with open(inputs['params'][elem], 'r') as f:
            tmp_symf = f.readlines()
            input_size = len(tmp_symf)
        if len(feature_list[elem]) < input_size:
            err = "Number of [{}] feature point[{}] is less than input size[{}]. This cause error during calculate PCA matirx".format(elem, len(feature_list[elem]), input_size)
            raise ValueError(err)

    pca = dict()
    scale_process = None

    for elem in inputs['atom_types']:
        pca_temp = PCA()
        scale_process = (feature_list[elem] - scale[elem][0].reshape(1, -1) )  / scale[elem][1].reshape(1, -1)
        pca_temp.fit(scale_process)
        min_level = inputs['preprocessing']['pca_min_whiten_level'] if inputs['preprocessing']['pca_min_whiten_level'] else 0.0
        # PCA transformation = x * pca[0] - pca[2] (divide by pca[1] if whiten)
        pca[elem] = [pca_temp.components_.T,
                     np.sqrt(pca_temp.explained_variance_ + min_level),
                     np.dot(pca_temp.mean_, pca_temp.components_.T)]

    
    return pca



def _calculate_gdf(inputs, logfile, feature_list_train, comm):
    #Call mpi if possible
    modifier = None

    if inputs['preprocessing']['weight_modifier']['type'] == 'modified sigmoid':
        modifier = dict()
        for item in inputs['atom_types']:
            modifier[item] = functools.partial(modified_sigmoid, **inputs['preprocessing']['weight_modifier']['params'][item])
    if inputs['preprocessing']['atomic_weights']['type'] == 'gdf':
        get_atomic_weights = _generate_gdf_file
    elif inputs['atomic_weights']['type'] == 'user':
        get_atomic_weights = user_atomic_weights_function
    elif inputs['atomic_weights']['type'] == 'file':
        get_atomic_weights = './atomic_weights'
    else:
        get_atomic_weights = None

    atomic_weights_train = atomic_weights_valid = None
    if callable(get_atomic_weights):
        # FIXME: use mpi
        local_target_list = dict()
        local_idx_list = dict()

        for item in self.parent.inputs['atom_types']:
            q = feature_list_train[item].shape[0] // comm.size
            r = feature_list_train[item].shape[0]  % comm.size

            begin = comm.rank * q + min(comm.rank, r)
            end = begin + q
            if r > comm.rank:
                end += 1

            local_target_list[item] = feature_list_train[item][begin:end]
            local_idx_list[item] = idx_list_train[item][begin:end]

        atomic_weights_train, dict_sigma, dict_c = get_atomic_weights(feature_list_train, scale, inputs['atom_types'], local_idx_list,\
         target_list=local_target_list, filename='atomic_weights', comm=comm, **kwargs)
        kwargs.pop('sigma')

        if comm.rank == 0:
            logfile.write('Selected(or generated) sigma and c\n')
            for item in inputs['atom_types']:
                logfile.write('{:3}: sigma = {:4.3f}, c = {:4.3f}\n'.format(item, dict_sigma[item], dict_c[item]))

        local_target_list = dict()
        local_idx_list = dict()

        for item in inputs['atom_types']:
            q = feature_list_valid[item].shape[0] // comm.size
            r = feature_list_valid[item].shape[0]  % comm.size

            begin = comm.rank * q + min(comm.rank, r)
            end = begin + q
            if r > comm.rank:
                end += 1

            local_target_list[item] = feature_list_valid[item][begin:end]
            local_idx_list[item] = idx_list_valid[item][begin:end]

        atomic_weights_valid, _, _      = get_atomic_weights(feature_list_train, scale, parent.inputs['atom_types'], local_idx_list,\
         target_list=local_target_list, sigma=dict_sigma, comm=comm, **kwargs)
    elif isinstance(get_atomic_weights, six.string_types):
        atomic_weights_train = pickle_load(get_atomic_weights)
        atomic_weights_valid = 'ones'

    if atomic_weights_train is None:
        aw_tag = False
    else:
        aw_tag = True
        #grp.plot_gdfinv_density(atomic_weights_train, self.parent.inputs['atom_types'])
        # Plot histogram only if atomic weights just have been calculated.
        if comm.rank == 0 and callable(get_atomic_weights):
            grp.plot_gdfinv_density(atomic_weights_train, inputs['atom_types'], auto_c=dict_c)
 
  

