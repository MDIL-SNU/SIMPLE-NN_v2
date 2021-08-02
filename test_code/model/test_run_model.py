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
from simple_nn_v2.models.run import run_model


from  simple_nn_v2.features.symmetry_function import generate
from  simple_nn_v2.features import preprocess

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)

generate(inputs, logfile)
print('Generate done')
preprocess(inputs, logfile)
print('Preprocess done')

run_model(inputs, logfile)
try:
    print('run_model test')
    print('rum_model OK')
    print('')
except:
    print('!!  Error occured run_model')
    print(sys.exc_info())
    print('')

