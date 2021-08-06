import sys
import os
import yaml
import functools
import atexit
import time 
#from simple_nn_v2.utils import modified_sigmoid, _generate_gdf_file
from ._version import __version__, __git_sha__

from simple_nn_v2.init_inputs import initialize_inputs, check_inputs
from simple_nn_v2.features import preprocess
from simple_nn_v2.models import train
from simple_nn_v2.features.symmetry_function.mpi import DummyMPI, MPI4PY

#from models import neural_network

#input parameter descriptor
def run(input_file_name):
    logfile = sys.stdout
    logfile = open('LOG', 'w', 10)
    atexit.register(_close_log, logfile)
    _log_header(logfile)

    #Load MPI 
    try: 
        comm = MPI4PY()
        assert comm.size != 1
        logfile.write("Use mpi with size {0}\n".format(comm.size))
    except:
        comm = DummyMPI()
        logfile.write("Not use mpi \n")

    inputs = initialize_inputs(input_file_name, logfile)
    if inputs['generate_features'] is True:
        start_time = time.time()
        if comm.rank == 0:
            check_inputs(inputs, logfile,'generate')
        generate = get_generate_function(logfile, descriptor_type=inputs['descriptor']['type'])
        generate(inputs, logfile, comm)
    
    if inputs['preprocess'] is True:
        if comm.rank == 0:
            check_inputs(inputs, logfile,'preprocess')
        preprocess(inputs, logfile, comm)

    if inputs['train_model'] is True:
        if comm.rank == 0:
            check_inputs(inputs, logfile,'train_model')
            train(inputs, logfile)
        else: 
            comm.free()

    if inputs['train_replica'] is True:
        pass


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

