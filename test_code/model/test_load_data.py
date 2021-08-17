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
from simple_nn_v2.models.run import _init_model, _load_data

# Setting for test preprocessing 

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)

model, optimizer, criterion, scale_factor, pca = _init_model(inputs, logfile)
print('_init_model called')
try:
    print('_load_data test')
    #Not fully tested yet!!
    scale_factor, pca, train_dataset, valid_dataset = _load_data(\
    inputs, logfile, model, optimizer, scale_factor, pca)
    print('_load_data OK')
    print('Train_dataset  : ', train_dataset)
    print('Valid_dataset  : ', valid_dataset)
    print('Scale factor   : ',scale_factor)
    print('PCA            : ',pca)
    print('')
except:
    print('!!  Error occured _load_data')
    print(sys.exc_info())
    print('')

