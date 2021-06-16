import re
from collections import OrderedDict
import torch
import numpy as np
from six.moves import cPickle as pickle

def load_data(data_file, pickle_format=False):
    if pickle_format:
        with open(data_file, 'rb') as fil:
            tmp_data = pickle.load(fil, encoding='latin1')
    else:
        tmp_data = torch.load(data_file)

    return tmp_data

def _gen_2Darray_for_ffi(arr, ffi, cdata='double'):
    # Function to generate 2D pointer for cffi  
    shape = arr.shape
    arr_p = ffi.new(cdata + " *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast(cdata + " *", arr[i].ctypes.data)
    return arr_p

def _make_full_featurelist(filelist, feature_tag, atom_types=None, pickle_format=False):
    """
    Args:
        filelist(str): name of file list ex) "pickle_train_list", "pickle_valid_list"
        feature_tag(str): key in data file that generated at generate process ex) 'x', 'dx', 'tot_num', ...
        atom_types
            None: atom type is not considered
            list: use atom_types list
        pickle_format(boolean): boolean if data file is .pickle format (else, .pt format)
    Returns:
        feature_list(dict): feature list of feature_tag
    """

    data_list = _make_data_list(filelist)
    feature_list = dict()
    
    if atom_types == None:
        for i,item in enumerate(data_list):
            tmp_data = load_data(item)
            feature_list.append(tmp_data[feature_tag])

        feature_list = np.concatenate(feature_list, axis=0)

    else:
        for item in atom_types:
            feature_list[item] = list()

        for i,item in enumerate(data_list):
            tmp_data = load_data(item)
            for jtem in atom_types:
                if jtem in tmp_data[feature_tag]:
                    feature_list[jtem].append(tmp_data[feature_tag][jtem])
            
        for item in atom_types:
            if len(feature_list[item]) > 0:
                feature_list[item] = np.concatenate(feature_list[item], axis=0)

    return feature_list

def _make_data_list(filename):
    return sum(_make_str_data_list(filename), [])

def _make_str_data_list(filename):
    """
    Read pickle_list file to make group of list of pickle files.
    Group id is denoted in front of the line (optional).
    If group id is not denoted, group id of -1 is assigned.

    example:
        0:file1.pickle
        0:file2.pickle
        14:file3.pickle
    """
    h = re.compile("([0-9]+):(.*)")
    data_list = OrderedDict()
    with open(filename, 'r') as fil:
        for line in fil:
            m = h.match(line.strip())
            if m:
                group_id, file_name = m.group(1), m.group(2)
            else:
                group_id, file_name = -1, line.strip()
            if group_id not in data_list:
                data_list[group_id] = []
            data_list[group_id].append(file_name)
    return data_list.values()
