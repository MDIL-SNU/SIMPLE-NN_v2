import sys
import os
import yaml
import collections
import functools
import atexit
from .utils import modified_sigmoid, _generate_gdf_file
from ._version import __version__, __git_sha__
from .utils.mpiclass import DummyMPI, MPI4PY
import tensorflow as tf
import numpy as np

#from features import symmetry_function
#from models import neural_network
from .init_inputs import initialize_inputs, _close_log, _log_header
    

#input parameter descriptor, model --> function 
def run(input_file_name, descriptor=None, model=None):
    # Set log file
    logfile = sys.stdout
    logfile = open('LOG', 'w', 10)
    atexit.register(_close_log)
    _log_header()

    # Initialize inputs
    inputs = initialize_inputs(input_file_name, logfile)
    '''
    # Set modifier, atomic weights
    modifier = None
    if inputs[descriptor]['weight_modifier']['type'] == 'modified sigmoid':
        modifier = dict()
        #modifier = functools.partial(modified_sigmoid, **self.descriptor.inputs['weight_modifier']['params'])
        for item in inputs['atom_types']:
            modifier[item] = functools.partial(modified_sigmoid, **inputs[descriptor]['weight_modifier']['params'][item])
    if inputs[descriptor]['atomic_weights']['type'] == 'gdf':
        #get_atomic_weights = functools.partial(_generate_gdf_file)#, modifier=modifier)
        get_atomic_weights = _generate_gdf_file
    elif inputs[descriptor]['atomic_weights']['type'] == 'user':
        get_atomic_weights = user_atomic_weights_function
    elif inputs[descriptor]['atomic_weights']['type'] == 'file':
        get_atomic_weights = './atomic_weights'
    else:
        get_atomic_weights = None

    # main running part
    if inputs['generate_features']:
        symmetry_function.generate(inputs, logfile)
    
    if inputs['preprocess']:
        symmetry_function.preprocess(inputs, logfile, get_atomic_weights=get_atomic_weights)

    if inputs['train_model']:
        neural_network.train(user_optimizer=user_optimizer, aw_modifier=modifier)
   '''
