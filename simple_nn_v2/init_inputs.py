import sys
import os
import yaml
import collections
import numpy as np
from .utils import DummyMPI
import torch
   
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
                'train_list':'./train_list', # (ADDED)
                'valid_list':'./valid_list', # (ADDED)
                'params': dict(),
                'refdata_format': 'vasp-out',
                'compress_outcar': True,
                'valid_rate': 0.1,
                'shuffle': True,
                'add_NNP_ref': False, # atom E to tfrecord
                'remain_pickle': False,
                'continue': False,
                'add_atom_idx': True, # For backward compatability
                'atomic_weights': {
                    'type': None,
                    'params': dict(),
                },
                'weight_modifier': {
                    'type': None,
                    'params': dict(),
                },
                'calc_scale': True, # calculation scale factor (ADDED)
                'scale_type': 'minmax',
                'scale_scale': 1.0,
                'scale_rho': None,

                'save_to_pickle': False, # default format is .pt / if True, format is .pickle (ADDED)
                'save_directory': './data', # directory of data files (ADDED)

                'read_force': True, #Read force in non-vasp files(ex. LAMMPS) (ADDED)
                'read_stress': True, #Read stress in non-vasp files(ex. LAMMPS) (ADDED)
                'dx_save_sparse': True,  # Save derivative tensor as sparse tensor (ADDED)
                'single_file': False   # Save all data into single file (ADDED)
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
                'regularization': 1e-6, #L2 regularization
                'use_force': True,
                'use_stress': False,
                'double_precision': True,
                'weight_initializer': {
                    'type': 'xavier normal',
                    'params': {

                    },
                },
                'acti_func': 'sigmoid',
                'dropout': None,

                # Optimization related
                'method': 'Adam',
                'batch_size': 64,
                'full_batch': False,
                'total_epoch': 10000,
                'total_iteration': None,  #Warning: Depreciated use total_epoch
                'learning_rate': 0.0001,
                'lr_decay': None,
                'stress_coeff': 0.000001,
                'force_coeff': 0.1, 
                'energy_coeff': 1.,
                'loss_scale': 1.,

                #pytorch 
                'workers': 4, # (ADDED)
                'force_diffscale':False,  # if true, force error is |F-F'|*(F-F')^2 instead of *(F-F')^2 (ADDED)

                # Loss function related
                'E_loss': 0,
                'F_loss': 1,

                # Logging & saving related (Epoch)
                'save_interval': 1000, 
                'show_interval': 100,
                'save_criteria': None,
                'save_best':False,
                'break_max': 10,
                'print_structure_rmse': False,

                #'cache': False,
                'pca': False,
                'pca_whiten': True,
                'pca_min_whiten_level': 1e-8,

                # Write atomic energies to pickle
                'NNP_to_pickle': False,


                #RESUME parameters (Temporary)
                'resume': None, #This should be directory of pytorch.save model file!! (ADDED)
                'clear_prev_status': False,  #Erase status before traning (ADDED)
                'clear_prev_network': False, #Erase network informateion  brfore training (ADDED)
                'start_epoch': 0 # if resume is not null(None), start_epoch is automatically set from checkpoint file


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
        torch.manual_seed(seed)
        np.random.seed(seed)
        logfile.write("*** Random seed: {0:} ***\n".format(seed))

    #Warning about epoch & iteration (epoch only avaialble)
    if inputs['neural_network']['total_iteration']:
        inputs['neural_network']['total_epoch'] = inputs['neural_network']['total_iteration']
        logfile.write("Warning: iteration is not available. Implicitly convert total_iteration to total_epoch\n")

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

def write_inputs(self):
    """
    Write current input parameters to the 'input_cont.yaml' file
    """
    with open('input_cont.yaml', 'w') as fil:
        yaml.dump(self.inputs, fil, default_flow_style=False)

