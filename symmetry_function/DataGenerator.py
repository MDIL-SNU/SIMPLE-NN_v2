import pickle
from braceexpand import braceexpand
import numpy as np
import ase
from ase import io
import os

class DataGenerator:
    def __init__(self, inputs, structure_list='./str_list', pickle_list='./pickle_list'):
        self.structure_list = structure_list
        self.pickle_list = pickle_list
        self.data_idx = 1
        self.train_dir = './data'
    
    def parse_structure_list(self):
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
                # Structure tag is in front of ":" and weight is back of ":" ex) [structur_tag : weight]
                # If no explicit weight(no ":"), default weight=1.0  ex) [structure_tag]
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
                    for item in list(braceexpand(tmp_split[0])):
                        structures.append([item, tmp_split[1]])
                        structure_idx.append(structure_tags.index(tag))
                except:  # what error?? # test code need
                    print('Unexpected line format in [str_list]')
                    print('ERROR LINE: {}'.format(line))
                    exit()

        return structures, structure_idx, structure_tags, structure_weights

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

    def load_snapshots(self, inputs, item, index):
        if inputs['refdata_format'] == 'vasp-out':
            if inputs['compress_outcar']:
                tmp_name = compress_outcar(item[0])

                if ase.__version__ >= '3.18.0':
                    snapshots = io.read(tmp_name, index=index, format=inputs['refdata_format'])
                else:
                    snapshots = io.read(tmp_name, index=index, format=inputs['refdata_format'], force_consistent=True)
            else:    
                if ase.__version__ >= '3.18.0':
                    snapshots = io.read(item[0], index=index, format=inputs['refdata_format'])
                else:
                    snapshots = io.read(item[0], index=index, format=inputs['refdata_format'], force_consistent=True)
        else:
            snapshots = io.read(item[0], index=index, format=inputs['refdata_format'])
        
        return snapshots

    def save_to_pickle(self, data, idx, save_dir='./data'):
        # ...If pickle_list not opened, open pickle_list first...


        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tmp_filename = os.path.join(save_dir, "data{}.pickle".format(self.data_idx))
        #tmp_filename = os.path.join(data_dir, "data{}.tfrecord".format(data_idx))

        # TODO: add tfrecord writing part
        #self._write_tfrecords(res, tmp_filename)
        with open(tmp_filename, "wb") as fil:
            pickle.dump(res, fil, protocol=2)  

        train_dir.write('{}:{}\n'.format(idx, tmp_filename))
        tmp_endfile = tmp_filename
        self.data_idx += 1

        # ...After finish iterative save, close pickle_list finally...

    def check_exist_data(self):
        pass

    def add_data(self):
        pass