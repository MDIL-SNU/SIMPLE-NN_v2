import sys
import os
sys.path.append('../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2.utils import _make_full_featurelist
from simple_nn_v2.init_inputs import  initialize_inputs
from simple_nn_v2.models.run import _init_model

# Setting for test preprocessing 

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)
print('Called input.yaml, logfile')

try:
    print('_init_model test')
    model, optimizer, criterion, scale_factor, pca = _init_model(inputs, logfile)
    print('_init_model OK')
    print('MODEL      : ',model)
    print('OPTIMIZER  : ',optimizer)
    print('CRITERION  : ',criterion)
    print('SCALE_FACTOR : ',scale_factor)
    print('PCA   : ',pca)
    print('')
except:
    print('!!  Error occured _init_model')
    print(sys.exc_info())
    print('')

