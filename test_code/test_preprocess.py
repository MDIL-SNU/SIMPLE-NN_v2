import sys
import os
sys.path.append('../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2 import Simple_nn
from simple_nn_v2.features.symmetry_function import Symmetry_function
from simple_nn_v2.models.neural_network import Neural_network
from simple_nn_v2.utils import _make_full_featurelist
from simple_nn_v2.utils.mpiclass import DummyMPI

os.system('rm -r ./data')

# Minimum Setting for Testing feature Descriptor class

model = Simple_nn('input.yaml', descriptor=Symmetry_function(), model=Neural_network() )
model.descriptor.set_inputs()
model.model.set_inputs()

print(model.model)

comm = DummyMPI()
os.system('rm -r ./data')
try:
    print('Symmetry_function.generate() test before preprocess')
    model.descriptor.generate()
    print('Symmetry_function.generate() OK')
    print('')
except:
    print('!!  Error occured in Symmetry_function.generate() ')
    print(sys.exc_info())
    print('')

try:
    print('Symmetry_function._split_data test')
    model.descriptor._split_data(comm)
    print('Symmetry_function._split_data OK')
    os.system('cat ./pickle_train_list')
    print('')
except:
    print('!!  Error occured in Symmetry_function._split_data ')
    print(sys.exc_info())
    print('')

try:
    print('Symmetry_function._calc_scale test')
    feature_list_train, idx_list_train = \
            _make_full_featurelist('./pickle_train_list', 'x', model.descriptor.parent.inputs['atom_types'], is_ptdata=True)
    print(feature_list_train.keys()) 
    scale = model.descriptor._calc_scale(True,feature_list_train, comm)
    print('Symmetry_function._calc_scale OK')
    print(scale)
    print('')
except:
    print('!!  Error occured in Symmetry_function._calc_scale ')
    print(sys.exc_info())
    print('')

try:
    print('Symmetry_function._generate_pca test')
    model.descriptor._generate_pca(feature_list_train, scale)
    print('Symmetry_function._generate_pca OK')
    print('')
except:
    print('!!  Error occured in Symmetry_function._generate_pca ')
    print(sys.exc_info())
    print('')












try:
    print('Symmetry_function.preprocess()')
    model.descriptor.preprocess()
    print('Symmetry_function.preprocess() OK')
    print('')
except:
    print('!!  Error occured in Symmetry_function.preprocess() ')
    print('')
    print(sys.exc_info())


