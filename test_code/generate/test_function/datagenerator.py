import sys
import os
sys.path.append('../../../')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#os.system('python libsymf_builder.py')

from simple_nn_v2 import Simple_nn
from simple_nn_v2.features.symmetry_function import Symmetry_function
from simple_nn_v2.utils.datagenerator import Data_generator

os.system('rm -r ./data')

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

model = Simple_nn('input.yaml', descriptor=Symmetry_function())
descriptor = Symmetry_function()
descriptor.parent = model
descriptor.set_inputs()
try:
    print('Checking parameters to use')
    print('INPUTS')
    for key in descriptor.inputs.keys():
        print('{0}    :'.format(key)  ,descriptor.inputs[key])
    print('')
    print('STRUCTURE_LIST   : ',descriptor.structure_list)
    print('PICKLE_LIST   :',descriptor.pickle_list)
    print('Checking parameters DONE')
    print('')
except:
    print('!!! Error occured during checking parameters')

try:
    print('Datagenerator.__init__() test')
    data_gen = Data_generator(descriptor.inputs, model.logfile, descriptor.structure_list, descriptor.pickle_list)
    print('inputs                :  ', data_gen.inputs)
    print('structure_list        :  ', data_gen.structure_list)
    print('pickle_list           :  ', data_gen.pickle_list)
    print('data_dir              :  ', data_gen.data_dir)
    print('_is_pickle_list_open  :  ', data_gen._is_pickle_list_open)
    print('_data_idx             :  ', data_gen._data_idx)
    print('parent                :  ', data_gen.parent)
    print('Datagenerator.__init__() OK')
    print('')
except:
    print('!!  Error occured in Datagenerator.__init__()')
    print('')

try:
    print('Datagenerator.parse_structure_list() test')
    structures, structure_tag_idx, structure_tags, structure_weights = data_gen.parse_structure_list()  
    print('structures   :',structures)
    print('structure_tag_idx   :',structure_tag_idx)
    print('structure_tags   :',structure_tags)
    print('structure_weights   :',structure_weights)
    print('Datagenerator.parse_structure_list() OK')
    print('')
except:
    print('!!  Error occured in Datagenerator.parse_strucrue_list()')
    print('')



try:
    print('Datagenerator._get_tag_and_weight() test')
    test_text = ['OUTCAR1','OUTCAR2:3','OUTCAR3 : 3']
    for text in test_text:
        print('Main text   :   '+text)
        tag , weight = data_gen._get_tag_and_weight(text)
        print('Tag         :  '+tag)
        print('Weight      :  ',weight)
        print('')
    print('Datagenerator._get_tag_and_weight() OK')
    print('')
except:
    print('!!  Error occured in Datagenerator._get_tang_and_weight()')
    print('')


try:
    print('Datagenerator.load_snapshots() test')
    test_struct = [['../../test_data/GST/OUTCAR_1', '::3'], ['../../test_data/GST/OUTCAR_2', '::'], ['../../test_data/GST/OUTCAR_3', '::']]
    ### utils.compress_outcar NOT working !!!
    ### ase.__version__ : '3.21.1' 
    ### ase.io.PareseError : Did not find required key "species" in parsed header result
    data_gen.inputs['compress_outcar'] =  False
    if data_gen.inputs['compress_outcar']: print('Generating ./tmp_comp_OUTCAR')
    for item in test_struct:
        print('Main structure  :   ',item)
        snapshot = data_gen.load_snapshots(item)
        print('Snapshot        :  ',snapshot)
        print('')
    print('Datagenerator.load_snapshots() OK')
    print('')
except:
    print('!!  Error occured in Datagenerator.load_snampshots()')
    print('')


try:
    print('Datagenerator.save_to_pickle() test')
    data = {'TMP':2 , 'DATA':5}
    tag_idx = 1
    print('Test dictrionary data ',data)
    print('Tag_index ',tag_idx)
    tmp_filename = data_gen.save_to_datafile(data, tag_idx)
    print('Tmp_filename    :',tmp_filename)
    print('Datagenerator.save_to_pickle() OK')
    print('')
except:
    print('!! Error occured in Datagenerator.save_to_pickle()')
    print('')



try:  ### Something strange for exising data..
    print('Datagenerator._check_exist_data() test')
    os.system('rm -r ./data')
    print('Test save_dir = ./data')
    print('Before _data_idx               : ',data_gen._data_idx)
    data_gen.save_to_datafile(data,tag_idx)
    print('Afert _data_idx (nonexist path)   :',data_gen._data_idx)
    os.system('touch ./data/data1.pt ./data/data2.pt ./data/data3.pt ./data/data4.pt')
    os.system('touch ./data/data5.pt ./data/data6.pt ./data/data7.pt ./data/data8.pt')
    print('8 Files in ./data')
    os.system('ls ./data')
    data_gen.save_to_datafile(data,tag_idx)
    print('Afert _data_idx (exist path):',data_gen._data_idx)
    os.system('rm -r ./data')
    print('Datagenerator._check_exist_data() OK')
    print('')
except:
    print('!! Error occured in Datagenerator._check_exist_data()')
    os.system('rm -r ./data')
    print('')

