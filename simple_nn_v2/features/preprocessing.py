import os, sys
import six
import numpy as np
import torch

from simple_nn_v2.utils import features as util_feature
from simple_nn_v2.utils import scale as util_scale

from sklearn.decomposition import PCA
 

def preprocess(inputs, logfile):
    """
    1. Split train/valid data names and save in "./pickle_train_list", "./pickle_valid_list" files
    2. Calculate scale factor with symmetry function values of train data set and save as "scale_factor" data (.pt format)
    3. Make PCA matrix with using scikitlearn modules

    Args:
        inputs(dict): full parts in input.yaml
        logfile(file obj): logfile object
    """ 

    data_list = './total_list'
    pickle_format = inputs['descriptor']['save_to_pickle']   #boolean
    make_pca_matrix = inputs['descriptor']['calc_pca']   # boolean

    _split_train_list_and_valid_list(inputs, data_list)

    # Extract specific feature values('x') from generated data files
    # feature_list.shape(): [(sum of atoms in each data file), (feature length)]
    train_feature_list = util_feature._make_full_featurelist(inputs['descriptor']['train_list'], 'x', inputs['atom_types'], pickle_format=pickle_format)
    #valid_feature_list = util_feature._make_full_featurelist(inputs['symmetry_function']['valid_list'], 'x', inputs['atom_types'], pickle_format=pickle_format)

    # scale[atom_type][0]: (mid_range or mean) of each features
    # scale[atom_type][1]: (width or standard_deviation) of each features
    scale = _calculate_scale(inputs, logfile, train_feature_list)
    torch.save(scale, 'scale_factor')

    # pca[atom_type][0]: principle axis matrix
    # pca[atom_type][1]: variance in each axis
    # pca[atom_type][2]: 
    if make_pca_matrix is True:
        pca = _calculate_pca_matrix(inputs, logfile, train_feature_list, scale)
        torch.save(pca, 'pca')

# Split train/valid data names that saved in data_list
def _split_train_list_and_valid_list(inputs, data_list='./total_list'):
   train_list_file = open(inputs['descriptor']['train_list'], 'w')
   valid_list_file = open(inputs['descriptor']['valid_list'], 'w')

   for file_list in util_feature._make_str_data_list(data_list):
       if inputs['descriptor']['shuffle'] is True:
           np.random.shuffle(file_list)
       num_pickle = len(file_list)
       num_valid = int(num_pickle * inputs['descriptor']['valid_rate'])

       for i,elem in enumerate(file_list):
           if i < num_valid:
               valid_list_file.write(elem + '\n')
           else:
               train_list_file.write(elem + '\n')
               
   train_list_file.close()
   valid_list_file.close()

# Calculate scale factor and save as "scale_factor" data (.pt format)
def _calculate_scale(inputs, logfile, feature_list):
    atom_types = inputs['atom_types']
    scale = None

    if not inputs['descriptor']['calc_scale']:
        scale = torch.load('./scale_factor')
    else:
        scale = dict()
        scale_type = inputs['descriptor']['scale_type']
        scale_scale = inputs['descriptor']['scale_scale']
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

                if logfile is not None:
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

        if logfile is not None:
            logfile.write("{:-^70}\n".format(""))

    return scale

# Make PCA matrix with using scikitlearn modules
def _calculate_pca_matrix(inputs, logfile, feature_list, scale):
    if inputs['neural_network']['pca']:
        for elem in inputs['atom_types']:
            with open(inputs['descriptor']['params'][elem], 'r') as f:
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
            min_level = inputs['neural_network']['pca_min_whiten_level'] if inputs['neural_network']['pca_min_whiten_level'] else 0.0
            # PCA transformation = x * pca[0] - pca[2] (divide by pca[1] if whiten)
            pca[elem] = [pca_temp.components_.T,
                         np.sqrt(pca_temp.explained_variance_ + min_level),
                         np.dot(pca_temp.mean_, pca_temp.components_.T)]

        logfile.write('PCA complete\n')
    
    return pca
