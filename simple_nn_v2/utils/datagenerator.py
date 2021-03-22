import pickle
from braceexpand import braceexpand
import numpy as np
import ase
from ase import io
import os
from . import compress_outcar

class Datagenerator:
    """ This class handles structure list file, OUTCAR files for making data set(format: pickle or torch)

    1. Parsing structure list file
    2. Loading structure snapshots in structure list
    3. After preprocess structure snapshots in Symmetry_function class, save results to pickle or torch files

    Attributes:
        inputs(dic): parsed dictionary form of [input.yaml] file
        structure_list(str): path of structure list file (default='./str_list') 
        pickle_list(str): path of file that contains every pickle file path (default='./pickle_list')
        data_dir(str): path of directory that contains pickle files
        _data_idx(int): index of current final pickle data (total number of pickle files in "data_dir")
    Methods:
        parse_structure_list(): Parsing "structure_list" file (default="./str_list")
        load_snapshots(): Read structure file and load snapshots using ase.io.read() method
        save_to_pickle(): 
        check_exist_data():
        add_data():
    """

    def __init__(self, inputs, structure_list='./str_list', pickle_list='./pickle_list', parent = None):
        self.inputs = inputs
        self.structure_list = structure_list
        self.pickle_list = pickle_list
        self.data_dir = './data'
        self._is_pickle_list_open = False
        self._data_idx = 0
        self.parent = parent

        #if inputs['add_data']:
        #    self.check_exist_data(self.inputs['add_data'])

    def __del__(self):
        if self._is_pickle_list_open == True:
            self._pickle_fil.close()
            self._is_pickle_list_open = False

    
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
                        self.parent.logfile.write("Error: {:}\n".format(err))
                        raise ValueError(err)
                    elif np.isclose(weight, 0):
                        self.parent.logfile.write("Warning: Structure weight for '{:}' is set to zero.\n".format(tag))

                    # If the same structure tags are given multiple times with different weights,
                    # other than first value will be ignored!
                    # Validate structure weight (structure weight is not validated on training run).
                    if tag not in structure_tags:
                        structure_tags.append(tag)
                        structure_weights.append(weight)    
                    else:                   
                        existent_weight = structure_weights[structure_tags.index(tag)]
                        if not np.isclose(existent_weight - weight, 0):
                            self.parent.logfile.write("Warning: Structure weight for '{:}' is set to {:} (previously set to {:}). New value will be ignored\n"\
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
                        print('Unexpected line format in [str_list]')
                        print('ERROR LINE: {}'.format(line))
                        exit()

        return structures, structure_tag_idx, structure_tags, structure_weights
    
    # Structure tag is in front of ":" and weight is back of ":" ex) [structur_tag : weight]
    # If no explicit weight(no ":"), default weight=1.0  ex) [structure_tag]
    @staticmethod
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
            self.parent.logfile.write('{} 0'.format(file_path))
        else:
            if ':' in item[1]:
                index = item[1]
            else:
                index = int(item[1])
            self.parent.logfile.write('{} {}'.format(file_path, item[1]))

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
            snapshots = io.read(file_path, index=index, format=self.inputs['refdata_format'])
        
        return snapshots

    def save_to_pickle(self, data, tag_idx, save_dir='./data'):
        # ... make input option : input['generate_features']['add_data'] ='./data'...
        """ Write result data to pickle file

        Check if pickle list file is open
        Check if previous data exist
        Save data as data{index}.pickle ({index}: +1 of last index in save directory)
        Write pickle file path in pickle list

        Args:
            data(dic): result data that created after preprocessing in Symmetry_function class
            tag_idx(int): structure tag index of snapshot
            save_dir(str): save directory for pickle file
        Returns:
            tmp_filename(str): saved pickle file path
        """

        if self._is_pickle_list_open == False:
            self._is_pickle_list_open = True
            self._pickle_fil = open(self.pickle_list, 'w')

        ### Error occur during this functon
        #self._check_exist_data(save_dir)
        # ADDED temprary
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._data_idx += 1
        try:
            tmp_filename = os.path.join(save_dir, "data{}.pickle".format(self._data_idx))
            with open(tmp_filename, "wb") as fil:
                pickle.dump(data, fil, protocol=2)
        except:
            self._data_idx -= 1
            # ...Raise error...

        self._pickle_fil.write('{}:{}\n'.format(tag_idx, tmp_filename))

        return tmp_filename

    # Check is "save_dir" exist
    # If previous pickle file exist, set _data_idx to last index of pickle for continue save
    def _check_exist_data(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            for fil in os.listdir(save_dir):
                last_idx = fil.split('.')[0][4:]
            self._data_idx = last_idx
