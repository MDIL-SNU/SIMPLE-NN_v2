import sys
import os
import yaml
import collections
import tensorflow as tf
import numpy as np
from .utils import DummyMPI
   
# init
def initialize_inputs(input_file_name, logfile):
    default_inputs = {
        'generate_features': True,
        'preprocess': False,
        'train_model': True,
        'atom_types': [],
        'random_seed': None,
        }
    descriptor_default_inputs = \
        {'symmetry_function': 
            {
                'params': dict(),
                'refdata_format': 'vasp-out',
                'compress_outcar': True,
                'data_per_tfrecord': 150,
                'valid_rate': 0.1,
                'shuffle': True,
                'add_NNP_ref': False, # atom E to tfrecord
                'remain_pickle': False,
                'continue': False,
                'add_atom_idx': True, # For backward compatability
                'num_parallel_calls': 5,
                'atomic_weights': {
                    'type': None,
                    'params': dict(),
                },
                'weight_modifier': {
                    'type': None,
                    'params': dict(),
                },
                'scale_type': 'minmax',
                'scale_scale': 1.0,
                'scale_rho': None,
                'save_to_pickle': False, # default format is .pt / if True, format is .pickle
                'save_directory': './data' # directory of data files
            }
        }
    model_default_inputs = \
        {'neural_network':
            {
                # Function related
                'train': True,
                'test': False,
                'continue': False,

                # Network related
                'nodes': '30-30',
                'regularization': {
                    'type': None,
                    'params': dict(),
                },
                'use_force': True,
                'use_stress': False,
                'double_precision': True,
                'weight_initializer': {
                    'type': 'truncated normal',
                    'params': {
                        'stddev': 0.3,
                    },
                },
                'acti_func': 'sigmoid',
                'dropout': None,

                # Optimization related
                'method': 'Adam',
                'batch_size': 64,
                'full_batch': False,
                'total_iteration': 10000,
                'learning_rate': 0.0001,
                'stress_coeff': 0.000001,
                'force_coeff': 0.1, 
                'energy_coeff': 1.,
                'loss_scale': 1.,
                'optimizer': dict(),

                # Loss function related
                'E_loss': 0,
                'F_loss': 1,

                # Logging & saving related
                'save_interval': 1000,
                'show_interval': 100,
                'save_criteria': [],
                'break_max': 10,
                'print_structure_rmse': False,

                # Performace related
                'inter_op_parallelism_threads': 0,
                'intra_op_parallelism_threads': 0,
                'cache': False,
                'pca': False,
                'pca_whiten': True,
                'pca_min_whiten_level': 1e-8,

                # Write atomic energies to pickle
                'NNP_to_pickle': False,
            }
        }
    
    # default inputs
    inputs = default_inputs
    inputs = _deep_update(inputs, descriptor_default_inputs)
    inputs = _deep_update(inputs, model_default_inputs)

    # update inputs using 'input.yaml'
    with open(input_file_name) as input_file:
        inputs = _deep_update(inputs, yaml.safe_load(input_file), warn_new_key=True,
                                    logfile=logfile)

    if len(inputs['atom_types']) == 0:
        raise KeyError

    if not inputs['neural_network']['use_force'] and \
            inputs['symmetry_function']['atomic_weights']['type'] is not None:
        logfile.write("Warning: Force training is off but atomic weights are given. Atomic weights will be ignored.\n")

    if inputs['neural_network']['method'] == 'L-BFGS' and \
            not inputs['neural_network']['full_batch']:
        logfile.write("Warning: Optimization method is L-BFGS but full batch mode is off. This might results bad convergence or divergence.\n")

    if inputs['random_seed'] is not None:
        seed = inputs['random_seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)
        logfile.write("*** Random seed: {0:} ***\n".format(seed))

    return inputs

def _deep_update(source, overrides, warn_new_key=False, logfile=None, depth=0, parent="top"):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.

    :param dict source: base dictionary to be updated
    :param dict overrides: new dictionary
    :param bool warn_new_key: if true, warn about new keys in overrides
    :param str logfile: filename to which warnings are written (if not given warnings are written to stdout)
    :returns: updated dictionary source
    """

    if logfile is None:
        logfile = sys.stdout

    for key in overrides.keys():
        if isinstance(source, collections.Mapping):
            if warn_new_key and depth < 2 and key not in source:
                logfile.write("Warning: Unidentified option in {:}: {:}\n".format(parent, key))
            if isinstance(overrides[key], collections.Mapping) and overrides[key]:
                returned = _deep_update(source.get(key, {}), overrides[key],
                                       warn_new_key=warn_new_key, logfile=logfile,
                                       depth=depth+1, parent=key)
                source[key] = returned
            # Need list append?
            else:
                source[key] = overrides[key]
        else:
            source = {key: overrides[key]}
    return source
