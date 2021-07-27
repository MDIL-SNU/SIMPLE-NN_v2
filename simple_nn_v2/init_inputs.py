import sys
import os
import yaml
import collections
import torch
import numpy as np

default_inputs = {
    'generate_features': True,
    'preprocess': True,
    'train_model': True,
    'atom_types': [],
    'random_seed': None,
}
symmetry_function_descriptor_default_inputs = \
        {'descriptor': 
            {
                'type': 'symmetry_function',
                'train_list': './train_list', 
                'valid_list': './valid_list', 
                'test_list': './test_list',

                'params': dict(),
                'refdata_format': 'vasp-out',
                'compress_outcar': True,

                'valid_rate': 0.1,
                'shuffle': True,

                'calc_scale': True, 
                'scale_type': 'minmax',
                'scale_scale': 1.0,
                'scale_rho': None,

                'save_to_pickle': False, 
                'save_directory': './data', 

                'read_force': True, #Read force in non-vasp files(ex. LAMMPS) Not implemented
                'read_stress': True, #Read stress in non-vasp files(ex. LAMMPS) Not implimented
                'dx_save_sparse': True, 

                'add_NNP_ref': False, # atom E 
                'add_atom_idx': True, # For backward compatability
                
                #Not implement yet
                'atomic_weights': {
                    'type': None,
                    'params': dict(),
                },
                'weight_modifier': {
                    'type': None,
                    'params': dict(),
                }
 
            }
        }
model_default_inputs = \
        {'neural_network':
            {
                # Function related
                'train': True,
                'test': False,
                
                # Network related
                'nodes': '30-30',
                'regularization': 1e-6, #L2 regularization
                'use_force': True,
                'use_stress': False,
                'double_precision': True,
                'weight_initializer': {
                    'type': 'xavier normal',
                    'params': {
                        'gain': None,
                        'std': None,
                        'mean': None,
                        'var': None
                    },
                },
                'acti_func': 'sigmoid',
                'dropout': None,

                # Optimization related
                'method': 'Adam',
                'batch_size': 64,
                'full_batch': False,
                'total_epoch': 1000,
                'total_iteration': None,  #Warning: Use total_epoch
                'learning_rate': 0.0001,
                'lr_decay': None,
                'stress_coeff': 0.000001,
                'force_coeff': 0.1, 
                'energy_coeff': 1.,
                'loss_scale': 1.,
                'optimizer': None,

                'workers': 0, 

                # Loss function related
                'E_loss_type': 0,
                'F_loss_type': 1,

                # Logging & saving related (Epoch)
                'save_interval': 100, 
                'show_interval': 10,
                'checkpoint_interval': None,
                'energy_criteria':None,
                'force_criteria':None,
                'stress_criteria':None,
                'break_max': 10,
                'print_structure_rmse': False,

                'pca': False,
                'pca_whiten': True,
                'pca_min_whiten_level': 1e-8,

                # Write atomic energies to pickle
                'NNP_to_pickle': False,
                'save_result': False,

                #RESUME parameters
                'continue': None, 
                'clear_prev_status': False,  
                'clear_prev_network': False,
                'start_epoch': 0,
                'read_potential':None,
                #Parallelism
                'inter_op_parallelism_threads': 0,
                'intra_op_parallelism_threads': 0,
                'load_data_to_gpu': False
            }
        }

def initialize_inputs(input_file_name, logfile):
    with open(input_file_name) as input_file:
        input_yaml = yaml.safe_load(input_file)
    descriptor_type = input_yaml['descriptor']['type']
    
    #set default inputs
    inputs = default_inputs
    descriptor_default_inputs = get_descriptor_default_inputs(logfile, descriptor_type=descriptor_type)
    inputs = _deep_update(inputs, descriptor_default_inputs)
    inputs = _deep_update(inputs, model_default_inputs)

    # update inputs using 'input.yaml'
    inputs = _deep_update(inputs, input_yaml, warn_new_key=True, logfile=logfile)

    if len(inputs['atom_types']) == 0:
        raise KeyError

    if not inputs['neural_network']['use_force'] and \
            inputs['descriptor']['atomic_weights']['type'] is not None:
        logfile.write("Warning: Force training is off but atomic weights are given. Atomic weights will be ignored.\n")

    if inputs['neural_network']['method'] == 'L-BFGS' and \
            not inputs['neural_network']['full_batch']:
        logfile.write("Warning: Optimization method is L-BFGS but full batch mode is off. This might results bad convergence or divergence.\n")

    if inputs['random_seed'] is not None:
        seed = inputs['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        logfile.write("*** Random seed: {0:} ***\n".format(seed))

    return inputs

def get_descriptor_default_inputs(logfile, descriptor_type='symmetry_function'):
    descriptor_inputs = {
        'symmetry_function': symmetry_function_descriptor_default_inputs
    }

    if descriptor_type not in descriptor_inputs.keys():
        err = "'{}' type descriptor is not implemented.".format(descriptor_type)
        logfile.write("\nError: {:}\n".format(err))
        raise NotImplementedError(err)

    return descriptor_inputs[descriptor_type]

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
