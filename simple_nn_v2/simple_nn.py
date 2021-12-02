import sys, os, time
import yaml, atexit
from ._version import __version__, __git_sha__

from simple_nn_v2.init_inputs import initialize_inputs, check_inputs
from simple_nn_v2.features import preprocess
from simple_nn_v2.models import train
from simple_nn_v2.features.mpi import DummyMPI, MPI4PY
from simple_nn_v2.features.symmetry_function import generate as symf_generator


def run(input_file_name):
    start_time = time.time()

    try:
        comm = MPI4PY()
        if comm.size == 1:
            comm = DummyMPI()
    except:
        comm = DummyMPI()

    logfile = sys.stdout
    if comm.rank == 0:
        logfile = open('LOG', 'w', 1)
        atexit.register(_close_log, logfile)
        _log_header(logfile)

    inputs = initialize_inputs(input_file_name, logfile, comm)

    if comm.size != 1 and inputs['train_model'] is True:
        if comm.rank == 0:
            err = "MPI4PY does not support in train model. Set train_model: False"
            logfile.write(err)
        raise NotImplementedError(err)
    else:
        if comm.rank == 0:
            logfile.write("MPI size {0}\n".format(comm.size))

    logfile.flush()

    if inputs['generate_features'] is True:
        errno = 0
        err = None
        if comm.rank == 0:
            errno, err = check_inputs(inputs, logfile, 'generate')
        check_errno(errno, err, comm)

        generate = get_generate_function(logfile, descriptor_type=inputs['descriptor']['type'])
        generate(inputs, logfile, comm)

    if inputs['preprocess'] is True:
        errno = 0
        err = None
        if comm.rank == 0:
            errno, err = check_inputs(inputs, logfile, 'preprocess')
        check_errno(errno, err, comm)

        preprocess(inputs, logfile, comm)

    if inputs['train_model'] is True:
        check_inputs(inputs, logfile, 'train_model')
        train(inputs, logfile)

    if comm.rank == 0:
        logfile.write(f"Total wall time {time.time()-start_time} s.\n")

def get_generate_function(logfile, descriptor_type='symmetry_function'):
    generator = {
        'symmetry_function': symf_generator
    }

    if descriptor_type not in generator.keys():
        err = "'{}' type descriptor is not implemented.".format(descriptor_type)
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

def check_errno(errno, err, comm):
    errno = comm.bcast(errno)
    err = comm.bcast(err)
    if errno != 0:
        raise Exception(err)

