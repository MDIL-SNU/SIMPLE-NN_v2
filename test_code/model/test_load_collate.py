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
from simple_nn_v2.models.run import _init_model, _load_data, _convert_to_tensor, _load_collate

# Setting for test preprocessing 

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)

model, optimizer, criterion, scale_factor, pca = _init_model(inputs, logfile)

scale_factor, pca, train_dataset, valid_dataset = _load_data(\
    inputs, logfile, model, optimizer, scale_factor, pca)

_convert_to_tensor(inputs, logfile, scale_factor, pca)

print('_convert_to_tensor called')
try:
    print('_load_collate test')
    train_loader, valid_loader = _load_collate(inputs, logfile, scale_factor, pca, train_dataset, valid_dataset)
    print('_load_collate OK')
    print('Train_loader  :', train_loader)
    print('Valid_loader  :', valid_loader)
    print('')
except:
    print('!!  Error occured _load_collate')
    print(sys.exc_info())
    print('')

