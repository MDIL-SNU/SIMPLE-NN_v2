import pickle
from braceexpand import braceexpand
import numpy as np
import ase
from ase import io
import os
from . import compress_outcar
import torch

class Data_generator:
    """ This class handles structure list file, OUTCAR files for making data set(format: pickle or torch)

    1. Parsing structure list file
    2. Loading structure snapshots in structure list
    3. After preprocess structure snapshots in Symmetry_function class, save results to pickle or pt files

    Attributes:
        inputs(dic): parsed dictionary form of [input.yaml] file
        structure_list(str): path of structure list file (default='./str_list') 
        data_list(str): path of file that contains every structure data file path (default='./pickle_list')
        _is_data_list_open(bool): check if data_list file is open
        _data_idx(int): index of current final pickle data (total number of pickle files in "data_dir")
        logfile(file stream): log file stream
        data_dir(str): directory path for saving data
    Methods:
        parse_structure_list(): Parsing "structure_list" file (default="./str_list")
        load_snapshots(): Read structure file and load snapshots using ase.io.read() method
        save_to_datafile(): 
    """

    def __init__(self, inputs, logfile, structure_list='./str_list', data_list='./pickle_list'):
        self.inputs = inputs
        self.structure_list = structure_list
        self.data_list = data_list
        self._is_data_list_open = False
        self._data_idx = 0
        self.logfile = logfile
        self.data_dir = inputs['save_directory']

        # check if data_dir is exist
        if os.path.exists(self.data_dir):
            err = "Directory {:} is already exist, remove directory or set another path in 'save_directory' option".format(self.data_dir)
            self.logfile.write("\nError: {:}\n".format(err))
            raise NotImplementedError(err)

    def __del__(self):
        if self._is_data_list_open == True:
            self._data_list_fil.close()
            self._is_data_list_open = False

    def parse_structure_list(self):
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

        Returns:
            structures: list of [STRUCTURE_PATH, INDEX_EXP]
            structure_tag_idx(int list): list of structure tag index of each STRUCTURE_PATH
            structure_tags(str list): list of structure tag
            structure_weights(float list): list of structure weight
        """
        structures = []
        structure_tag_idx = []
        structure_tags = ["None"]
        structure_weights = [1.0]
        tag = "None"
        weight = 1.0

        with open(self.structure_list, 'r') as fil:
            for line in fil:
                line = line.strip()

                if len(line) == 0 or line.isspace():
                    continue
                # 1. Extract structure tag and weight in between "[ ]"
                elif line[0] == "[" and line[-1] == "]":
                    tag_line = line[1:-1]
                    tag, weight = self._get_tag_and_weight(tag_line)
                    
                    if weight < 0:
                        err = "Structure weight must be greater than or equal to zero."
                        self.logfile.write("Error: {:}\n".format(err))
                        raise ValueError(err)
                    elif np.isclose(weight, 0):
                        self.logfile.write("Warning: Structure weight for '{:}' is set to zero.\n".format(tag))

                    # If the same structure tags are given multiple times with different weights,
                    # other than first value will be ignored!
                    # Validate structure weight (structure weight is not validated on training run).
                    if tag not in structure_tags:
                        structure_tags.append(tag)
                        structure_weights.append(weight)    
                    else:                   
                        existent_weight = structure_weights[structure_tags.index(tag)]
                        if not np.isclose(existent_weight - weight, 0):
                            self.logfile.write("Warning: Structure weight for '{:}' is set to {:} (previously set to {:}). New value will be ignored\n"\
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
                        self.logfile.write("\nError: {:}\n".format(err))
                        self.logfile.write("ERROR LINE: {:}\n".format(line))
                        raise NotImplementedError(err)

        return structures, structure_tag_idx, structure_tags, structure_weights
    
    # Structure tag is in front of ":" and weight is back of ":" ex) [structur_tag : weight]
    # If no explicit weight(no ":"), default weight=1.0  ex) [structure_tag]
    def _get_tag_and_weight(self, text):
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

    def load_snapshots(self, item):
        """ Read structure file and load snapshots using ase.io.read() method

        Handle inputs['refdata_format']: 'vasp-out' vs else
        Handle inputs['compress_outcar']: True vs False
        Handle ase version: later than 3.18.0 vs else

        Args:
            item: [STRUCTURE_PATH, INDEX]    ex) ["/path1/OUTCAR", "::10"] or ["/path2/OUTCAR", "10"]]
        Returns:
            snapshots(ase.atoms.Atoms object): Atoms object from ase module that contain structure information, E, F, S ...
        """
        file_path = item[0]
        if len(item) == 1:
            index = 0
            self.logfile.write('{} 0'.format(file_path))
        else:
            if ':' in item[1]:
                index = item[1]
            else:
                index = int(item[1])
            self.logfile.write('{} {}'.format(file_path, item[1]))

        if self.inputs['refdata_format'] == 'vasp-out':
            if self.inputs['compress_outcar']:
                tmp_name = compress_outcar(file_path)
                print(tmp_name)
                if ase.__version__ >= '3.18.0':
                    snapshots = io.read(tmp_name, index=index, format=self.inputs['refdata_format'])
                else:
                    snapshots = io.read(tmp_name, index=index, format=self.inputs['refdata_format'], force_consistent=True)
            else:    
                if ase.__version__ >= '3.18.0':
                    snapshots = io.read(file_path, index=index, format=self.inputs['refdata_format'])
                else:
                    snapshots = io.read(file_path, index=index, format=self.inputs['refdata_format'], force_consistent=True)
        else:
            self.logfile.write("Warning: Structure format is not OUTCAR(['refdata_format'] : {:}). Unexpected error can occur.\n"\
                                                        .format(self.inputs['refdata_format']))
            snapshots = io.read(file_path, index=index, format=self.inputs['refdata_format'])
        
        return snapshots

    def save_to_datafile(self, data, tag_idx):
        """ Write result data to pickle file

        Check if pickle list file is open
        Save data as data{index}.pickle ({index}: +1 of last index in save directory)
        Write pickle file path in pickle list

        Args:
            data(dic): result data that created after preprocessing in Symmetry_function class
            tag_idx(int): structure tag index of snapshot
        Returns:
            tmp_filename(str): saved pickle file path
        """

        if self._is_data_list_open == False:
            self._data_list_fil = open(self.data_list, 'w')
            self._is_data_list_open = True

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self._data_idx += 1
        try:
            if self.inputs['save_to_pickle'] == False:
                tmp_filename = os.path.join(self.data_dir, "data{}.pt".format(self._data_idx))
                torch.save(data, tmp_filename)
            elif self.inputs['save_to_pickle'] == True:
                tmp_filename = os.path.join(self.data_dir, "data{}.pickle".format(self._data_idx))
                with open(tmp_filename, "wb") as fil:
                    pickle.dump(data, fil, protocol=2)
        except:
            self._data_idx -= 1
            if self.inputs['save_to_pickle'] == False:
                err = "Unexpected error during save data to .pt file"
            else:
                err = "Unexpected error during save data to .pickle file"
            self.logfile.write("\nError: {:}\n".format(err))
            raise NotImplementedError(err)

        self._data_list_fil.write('{}:{}\n'.format(tag_idx, tmp_filename))

        return tmp_filename
