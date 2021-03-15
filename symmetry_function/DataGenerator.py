import pickle
from braceexpand import braceexpand
import numpy as np
import ase
from ase import io
import os
from ...utils import compress_outcar

class DataGenerator:
    """

    This class handles individual pickle data

    Attributes:
        structure_list: 
        pickle_list:
        data_idx:
        train_dir:
    Methods:
        parse_structure_list():
        load_snapshots():
        save_to_pickle():
        check_exist_data():
        add_data():
    """
    def __init__(self, inputs, structure_list='./str_list', pickle_list='./pickle_list'):
        self.inputs = inputs
        self.structure_list = structure_list
        self.pickle_list = pickle_list
        self._data_idx = 1
        self.train_dir = './data'
        
        if inputs['add_data']:
            self.check_exist_data(self.inputs['add_data'])
    
    def parse_structure_list(self):
        """ Parsing "structure_list" file (default="./str_list")

        ex) "str_list" file format:
                [structure_tag1 : weight]
                FILE_PATH1 INDEX1
                FILE_PATH2 INDEX2
                ...

                [structure_tag2 : weight]
                FILE_PATH3 INDEX3
                FILE_PATH4 INDEX4
                ...

        1. Seperate structure_tag and wieght in between "[ ]"
        2. Extract FILE_PATH and INDEX
        3. Set tag index for each FILE_PATH
        We support FILE_PATH as brace expansion ex) /path{1..10}/OUTCAR

        Returns:
            structures: 
            structure_idx:
            structure_tags:
            structure_weights:

        """
        structures = []
        structure_idx = []
        structure_tags = ["None"]
        structure_weights = [1.0]
        tag = "None"
        weight = 1.0

        with open(self.structure_list, 'r') as fil:
            for line in fil:
                line = line.strip()

                # 1. Extract structure tag and weight in between "[ ]"
                if len(line) == 0 or line.isspace():
                    continue
                elif line[0] == "[" and line[-1] == "]":
                    text = line[1:-1]
                    tag, weight = self._get_tag_and_weight(text)
                    
                    if weight < 0:
                        err = "Structure weight must be greater than or equal to zero."
                        self.parent.logfile.write("Error: {:}\n".format(err))
                        raise ValueError(err)
                    elif np.isclose(weight, 0) and self.comm.rank == 0:
                        self.parent.logfile.write("Warning: Structure weight for '{:}' is set to zero.\n".format(tag))

                    # If the same structure tags are given multiple times with different weights,
                    # other than first value will be ignored!
                    # Validate structure weight (structure weight is not validated on training run).
                    if tag not in structure_tags:
                        structure_tags.append(tag)
                        structure_weights.append(weight)    
                    else:                   
                        existent_weight = structure_weights[structure_tags.index(tag)]
                        if not np.isclose(existent_weight - weight, 0) and self.comm.rank == 0:
                            self.parent.logfile.write("Warning: Structure weight for '{:}' is set to {:} (previously set to {:}). New value will be ignored\n".format(tag, weight, old_weight))
                        continue

                # 2. Extract file name and range and set tag index

                tmp_split = line.split()
                try:
                    structures_expression = tmp_split[0]
                    step_ranges = tmp_split[1]
                    for item in list(braceexpand(structures_expression)):
                        structures.append([item, step_ranges])
                        structure_idx.append(structure_tags.index(tag))
                except:  # what error?? # test code need
                    print('Unexpected line format in [str_list]')
                    print('ERROR LINE: {}'.format(line))
                    exit()

        return structures, structure_idx, structure_tags, structure_weights
    
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

    def load_snapshots(self, item):   # ...remove input argument in __init__.py...
        """ Read structure file using ase.io.read() method

        Handle inputs['refdata_format']: 'vasp-out' vs else
        Handle inputs['compress_outcar']: True vs False
        Handle ase version: later than 3.18.0 vs else

        Args:
            item: [file_path, index]    ex) ["/path1/OUTCAR", "::10"] or ["/path2/OUTCAR", "10"]]

        Returns:
            snapshots: list of ase.atoms.Atoms object
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

    def save_to_pickle(self, data, idx, save_dir='./data'):
        # ... make input option : input['generate_features']['add_data'] ='./data'...
        """ Read structure file using ase.io.read() method

        H

        Args:
            data:
            idx:
            save_dir:
        """


        # ...If pickle_list not opened, open pickle_list first...


        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tmp_filename = os.path.join(save_dir, "data{}.pickle".format(self.data_idx))
        #tmp_filename = os.path.join(data_dir, "data{}.tfrecord".format(data_idx))

        # TODO: add tfrecord writing part
        #self._write_tfrecords(res, tmp_filename)
        with open(tmp_filename, "wb") as fil:
            pickle.dump(data, fil, protocol=2)  

        train_dir.write('{}:{}\n'.format(idx, tmp_filename))
        tmp_endfile = tmp_filename
        self._data_idx += 1

        # ...After finish iterative save, close pickle_list finally...

    def check_exist_data(self, save_dir):
        pass

    def add_data(self):
        pass