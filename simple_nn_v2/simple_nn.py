import sys
import os
import yaml
import functools
import atexit
#from simple_nn_v2.utils import modified_sigmoid, _generate_gdf_file
from ._version import __version__, __git_sha__

from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features import preprocess
from simple_nn_v2.models import train_NN 

#from models import neural_network

#input parameter descriptor
def run(input_file_name):
    logfile = sys.stdout
    logfile = open('LOG', 'w', 10)
    atexit.register(_close_log, logfile)
    _log_header(logfile)

    inputs = initialize_inputs(input_file_name, logfile)

    if inputs['generate_features'] is True:
        generate = get_generate_function(logfile, descriptor_type=inputs['descriptor']['type'])
        generate(inputs, logfile)
    
    if inputs['preprocess'] is True:
        preprocess(inputs, logfile)

    if inputs['train_model']:
        train_NN(inputs, logfile)

from simple_nn_v2.features.symmetry_function import generate as symf_generator

def get_generate_function(logfile, descriptor_type='symmetry_function'):
    generator = {
        'symmetry_function': symf_generator
    }

    if descriptor_type not in generator.keys():
        err = "'{}' type descriptor is not implemented.".format(descriptor_type)
        logfile.write("\nError: {:}\n".format(err))
        raise NotImplementedError(err)

    return generator[descriptor_type] 

def _close_log(logfile):
    logfile.flush()
    os.fsync(logfile.fileno())
    logfile.close()

def _log_header(logfile):
    # TODO: make the log header (low priority)
    logfile.write("SIMPLE_NN v{0:} ({1:})\n".format(__version__, __git_sha__))
    logfile.write("{}\n".format('-'*94))

    logfile.write("{:^94}\n".format("  _____ _ _      _ _ ___  _     _____       __    _ __    _"))
    logfile.write("{:^94}\n".format(" / ____| | \    / | '__ \| |   |  ___|     |  \  | |  \  | |"))
    logfile.write("{:^94}\n".format("| |___ | |  \  /  | |__) | |   | |___  ___ |   \ | |   \ | |"))
    logfile.write("{:^94}\n".format(" \___ \| |   \/   |  ___/| |   |  ___||___|| |\ \| | |\ \| |"))
    logfile.write("{:^94}\n".format(" ____| | | |\  /| | |    | |___| |___      | | \   | | \   |"))
    logfile.write("{:^94}\n".format("|_____/|_|_| \/ |_|_|    |_____|_____|     |_|  \__|_|  \__|"))

    logfile.write("{:^94}\n".format("                                                    ver2.0.0"))
    logfile.write("{}\n\n".format('-'*94))

def write_inputs(inputs):
    """
    Write current input parameters to the 'input_cont.yaml' file
    """
    with open('input_cont.yaml', 'w') as fil:
        yaml.dump(inputs, fil, default_flow_style=False)

