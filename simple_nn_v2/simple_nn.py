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

from features.symmetry_function import generating
#from models import neural_network
from init_inputs import initialize_inputs
    

#input parameter descriptor, model --> function 
def run(input_file_name, descriptor=None, model=None):
    # Set log file
    logfile = sys.stdout
    logfile = open('LOG', 'w', 10)
    atexit.register(_close_log, logfile)
    _log_header(logfile)

    # Initialize inputs
    inputs = initialize_inputs(input_file_name, logfile)
    
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
        if descriptor == 'symmetry_function':
            generating.generate(inputs, logfile)
    
    if inputs['preprocess']:
        if descriptor == 'symmetry_function':
            symmetry_function.preprocess(inputs, logfile, get_atomic_weights=get_atomic_weights)

#    if inputs['train_model']:
#        if model == 'neural_network':
#            neural_network.train(user_optimizer=user_optimizer, aw_modifier=modifier)


def _close_log(logfile):
    logfile.flush()
    os.fsync(logfile.fileno())
    logfile.close()

def _log_header(logfile):
    # TODO: make the log header (low priority)
    logfile.write("SIMPLE_NN v{0:} ({1:})\n".format(__version__, __git_sha__))

def write_inputs(inputs):
    """
    Write current input parameters to the 'input_cont.yaml' file
    """
    with open('input_cont.yaml', 'w') as fil:
        yaml.dump(inputs, fil, default_flow_style=False)

