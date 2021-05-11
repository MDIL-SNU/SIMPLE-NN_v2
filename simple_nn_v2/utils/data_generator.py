import pickle
from braceexpand import braceexpand
import numpy as np
import ase
from ase import io
import os
from . import compress_outcar
import torch

""" In this module, functions handles structure list file and OUTCAR files for making data set(format: pickle or torch)

    parse_structure_list(logfile, structure_list): Parsing "structure_list" file (default="./str_list")
    load_snapshots(inputs, item, logfile): Read structure file and load snapshots using ase.io.read() method
    save_to_datafile(inputs, data, data_idx, logfile): save results to pickle or pt files
"""

def parse_structure_list(logfile, structure_list='./str_list'):
    """ Parsing "structure_list" file (default="./str_list")

    ex) "str_list" file format:
            [structure_tag1 : weight]
            STRUCTURE_PATH_EXP1 INDEX_EXP1   ex) /PATH/OUTCAR1 ::10
            STRUCTURE_PATH_EXP2 INDEX_EXP2   ex) /PATH/OUTCAR2 10:1000:50
            ...

            [structure_tag2]
            STRUCTURE_PATH_EXP3 INDEX_EXP3   ex) /PATH/OUTCAR{1..10} ::
            STRUCTURE_PATH_EXP4 INDEX_EXP4   ex) /PATH{1..10}/OUTCAR ::
            ...

    1. Seperate structure_tag and weight in between "[ ]"
    2. Extract STRUCTURE_PATH from STRUCTURE_PATH_EXP and INDEX_EXP
    3. Set tag index for each STRUCTURE_PATH
    We support STRUCTURE_PATH_EXP as brace expansion ex) /PATH{1..10}/OUTCAR or /PATH/OUTCAR{1..10}

    Args:
        logfile(file obj): logfile object
        structure_list(str): name of structure list file that we have to parse
    Returns:
        structures: list of [STRUCTURE_PATH, INDEX_EXP]
        structure_tag_idx(int list): list of structure tag index of each STRUCTURE_PATH
        structure_tags(str list): list of structure tag
        structure_weights(float list): list of structure weight
    """

    structures = []
    structure_tag_idx = []
    structure_tags = ['None']
    structure_weights = [1.0]
    tag = 'None'
    weight = 1.0

    with open(structure_list, 'r') as fil:
        for line in fil:
            line = line.strip()

            if len(line) == 0 or line.isspace():
                continue
            # 1. Extract structure tag and weight in between "[ ]"
            elif line[0] == '[' and line[-1] == ']':
                tag_line = line[1:-1]
                tag, weight = _get_tag_and_weight(tag_line)
                
                if weight < 0:
                    err = "Structure weight must be greater than or equal to zero."
                    logfile.write("Error: {:}\n".format(err))
                    raise ValueError(err)
                elif np.isclose(weight, 0):
                    logfile.write("Warning: Structure weight for '{:}' is set to zero.\n".format(tag))

                # If the same structure tags are given multiple times with different weights,
                # other than first value will be ignored!
                # Validate structure weight (structure weight is not validated on training run).
                if tag not in structure_tags:
                    structure_tags.append(tag)
                    structure_weights.append(weight)    
                else:                   
                    existent_weight = structure_weights[structure_tags.index(tag)]
                    if not np.isclose(existent_weight - weight, 0):
                        logfile.write("Warning: Structure weight for '{:}' is set to {:} (previously set to {:}). New value will be ignored\n"\
                                                    .format(tag, weight, existent_weight))
                continue
            # 2. Extract STRUCTURE_PATH and INDEX, then set structure tag index
            else:
                tmp_split = line.split()
                try:
                    structure_path_exp = tmp_split[0]
                    index_exp = tmp_split[1]
                    for structure_path in list(braceexpand(structure_path_exp)):
                        structures.append([structure_path, index_exp])
                        structure_tag_idx.append(structure_tags.index(tag))
                except:
                    err = "Unexpected line format in [str_list]"
                    logfile.write("\nError: {:}\n".format(err))
                    logfile.write("ERROR LINE: {:}\n".format(line))
                    raise NotImplementedError(err)

    return structures, structure_tag_idx, structure_tags, structure_weights

# Structure tag is in front of ":" and weight is back of ":" ex) [structur_tag : weight]
# If no explicit weight(no ":"), default weight=1.0  ex) [structure_tag]
def _get_tag_and_weight(text):
    if ':' in text:
        splited_text = text.rsplit(':', 1)
        try:
            tag = splited_text[0].strip()
            weight = float(splited_text[1].strip())
        except ValueError:
            tag = text.strip()
            weight = 1.0
    else:
        tag = text.strip()
        weight = 1.0
        
    return tag, weight

def load_snapshots(inputs, item, logfile):
    """ Read structure file and load snapshots using ase.io.read() method

    Handle inputs['refdata_format']: 'vasp-out' vs else
    Handle inputs['compress_outcar']: True vs False
    Handle ase version: later than 3.18.0 vs else

    Args:
        inputs(dict): ['symmetry_function'] part in input.yaml
        item(list): [STRUCTURE_PATH, INDEX]    ex) ["/path1/OUTCAR", "::10"] or ["/path2/OUTCAR", "10"]]
        logfile(file obj): logfile object
    Returns:
        snapshots(ase.atoms.Atoms object): Atoms object from ase module that contain structure information, E, F, S ...
    """
    file_path = item[0]
    if len(item) == 1:
        index = 0
        logfile.write("{} 0".format(file_path))
    else:
        if ':' in item[1]:
            index = item[1]
        else:
            index = int(item[1])
        logfile.write("{} {}".format(file_path, item[1]))

    if inputs['refdata_format'] == 'vasp-out':
        if inputs['compress_outcar']:
            tmp_name = compress_outcar(file_path)

            if ase.__version__ >= '3.18.0':
                snapshots = io.read(tmp_name, index=index, format=inputs['refdata_format'])
            else:
                snapshots = io.read(tmp_name, index=index, format=inputs['refdata_format'], force_consistent=True)
        else:    
            if ase.__version__ >= '3.18.0':
                snapshots = io.read(file_path, index=index, format=inputs['refdata_format'])
            else:
                snapshots = io.read(file_path, index=index, format=inputs['refdata_format'], force_consistent=True)
    else:
        logfile.write("Warning: Structure format is not OUTCAR(['refdata_format'] : {:}). Unexpected error can occur.\n"\
                                                    .format(inputs['refdata_format']))
        snapshots = io.read(file_path, index=index, format=inputs['refdata_format'])
    
    return snapshots

def save_to_datafile(inputs, data, data_idx, logfile):
    """ Write result data to pickle file

    Check if pickle list file is open
    Save data as data{index}.pickle ({index}: +1 of last index in save directory)
    Write pickle file path in pickle list

    Args:
        inputs(dict): ['symmetry_function'] part in input.yaml
        data(dic): result data that contains information after calculting symmetry function values
        data_idx(int): index of data file to save
        logfile(file obj): logfile object
    Returns:
        tmp_filename(str): saved pickle file path
    """
    data_dir = inputs['save_directory']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        if inputs['save_to_pickle'] == False:
            tmp_filename = os.path.join(data_dir, 'data{}.pt'.format(data_idx))
            torch.save(data, tmp_filename)
        elif inputs['save_to_pickle'] == True:
            tmp_filename = os.path.join(data_dir, 'data{}.pickle'.format(data_idx))
            with open(tmp_filename, 'wb') as fil:
                pickle.dump(data, fil, protocol=2)
    except:
        if inputs['save_to_pickle'] == False:
            err = "Unexpected error during save data to .pt file"
        else:
            err = "Unexpected error during save data to .pickle file"
        logfile.write("\nError: {:}\n".format(err))
        raise NotImplementedError(err)

    return tmp_filename
