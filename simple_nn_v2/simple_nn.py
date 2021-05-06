import sys
import os
import yaml
import functools
import atexit
from .utils import modified_sigmoid, _generate_gdf_file
from ._version import __version__, __git_sha__
import numpy as np

from .init_inputs import initialize_inputs

from .features import preprocess as prep
from .features.symmetry_function import generate
from .models import train_NN
from .init_inputs import initialize_inputs

#input parameter descriptor, model --> function 
def run(input_file_name, descriptor=None, preprocess=None, model=None):
    # Set log file
    logfile = sys.stdout
    logfile = open('LOG', 'w', 10)
    atexit.register(_close_log, logfile)
    _log_header(logfile)

    # Initialize inputs
    inputs = initialize_inputs(input_file_name, logfile)
    
    # Set modifier, atomic weights
    modifier = None
    if inputs['symmetry_function']['weight_modifier']['type'] == 'modified sigmoid':
        modifier = dict()
        #modifier = functools.partial(modified_sigmoid, **self.descriptor.inputs['weight_modifier']['params'])
        for item in inputs['atom_types']:
            modifier[item] = functools.partial(modified_sigmoid, **inputs['symmetry_function']['weight_modifier']['params'][item])
    if inputs['symmetry_function']['atomic_weights']['type'] == 'gdf':
        #get_atomic_weights = functools.partial(_generate_gdf_file)#, modifier=modifier)
        get_atomic_weights = _generate_gdf_file
    elif inputs['symmetry_function']['atomic_weights']['type'] == 'user':
        get_atomic_weights = user_atomic_weights_function
    elif inputs['symmetry_function']['atomic_weights']['type'] == 'file':
        get_atomic_weights = './atomic_weights'
    else:
        get_atomic_weights = None

    get_atomic_weights = None
    if not descriptor:
        descriptor = generate
    if not preprocess:
        preprocess = prep
    if not model:
        model = train_NN
    # main running part
    if inputs['generate_features']:
        descriptor(inputs, logfile)

    if inputs['preprocess']:
        preprocess(inputs, logfile, get_atomic_weights=get_atomic_weights)

    #TODO:  should add atomic weight modifier 
    if inputs['train_model']:
        model(inputs, logfile) 




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

