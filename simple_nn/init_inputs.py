import sys
import os
import yaml
import collections
import torch
import numpy as np
import time

default_inputs = {
    'generate_features': True,
    'preprocess': True,
    'train_model': True,
    'random_seed': None,
    'params': dict(),
}

symmetry_function_data_default_inputs = \
        {'data':
            {
                'type'          :   'symmetry_function',
                'struct_list'   :   './structure_list',
                'refdata_format':   'vasp-out',
                'compress_outcar':  True,
                'save_directory':   './data',
                'save_list'     :   './total_list',
                'absolute_path' :   True,
                'read_force'    :   True,
                'read_stress'   :   True,
                'dx_save_sparse':   True,
            }
        }

preprocess_default_inputs = \
        {'preprocessing':
            {
                'data_list' : './total_list',
                'train_list': './train_list',
                'valid_list': './valid_list',
                'valid_rate': 0.1,
                'shuffle'   : True,
                #Scaling parameters
                'calc_scale': True,
                'scale_type': 'minmax',
                'scale_width': 1.0,
                'scale_rho' : None,
                #PCA parameters
                'calc_pca'  : True,
                'pca_whiten': True,
                'min_whiten_level': 1.0e-8,
                #Atomic weights
                'calc_atomic_weights': False,
            }
        }

model_default_inputs = \
        {'neural_network':
            {
                'train_list': './train_list',
                'valid_list': './valid_list',
                'test_list' : './test_list',
                'ref_list'  : './ref_list',

                'train'     : True,
                'test'      : False,
                'use_force' : True,
                'use_stress': True,
                'add_NNP_ref'   : False,
                'train_atomic_E': False,
                'test_atomic_E': False,
                'shuffle_dataloader': True,

                # Network related
                'nodes'     : '30-30',
                'acti_func' : 'sigmoid',
                'double_precision'  : True,
                'weight_initializer': {
                    'type'  : 'xavier normal',
                    'params': {
                        'gain'  : None,
                        'std'   : None,
                        'mean'  : None,
                        'val'   : None,
                        'sparsity':None,
                        'mode'  : None,
                        'nonlinearity': None,
                    },
                },
                'dropout'   : 0.0,
                'use_pca'   : True,
                'use_scale' : True,
                'use_atomic_weights'   : False,
                'weight_modifier': {
                    'type'  : None,
                    'params': dict(),
                },
                # Optimization
                'optimizer' : {
                    'method': 'Adam',
                    'params':
                        None
                },
                'batch_size'    : 8,
                'full_batch'    : False,
                'total_epoch'   : 1000,
                'learning_rate' : 0.0001,
                'decay_rate'    : None,
                'l2_regularization': 1.0e-6,
                # Loss function
                'loss_scale'    : 1.,
                'E_loss_type'   : 1,
                'F_loss_type'   : 1,
                'energy_coeff'  : 1.,
                'force_coeff'   : 0.1,
                'stress_coeff'  : 0.000001,
                # Logging & saving
                'show_interval' : 10,
                'save_interval':  0,
                'energy_criteria'   :   None,
                'force_criteria'    :   None,
                'stress_criteria'   :   None,
                'break_max'         :   10,
                'print_structure_rmse': False,
                'accurate_train_rmse':  True,
                # Restart
                'continue'      : None,
                'start_epoch'   : 1,
                'clear_prev_status'     : False,
                'clear_prev_optimizer'  : False,
                # Parallelism
                'use_gpu': True,
                'GPU_number'       : None,
                'subprocesses'   : 0,
                'inter_op_threads': 0,
                'intra_op_threads': 0,
            }
        }

def initialize_inputs(input_file_name, logfile):
    with open(input_file_name) as input_file:
        input_yaml = yaml.safe_load(input_file)
    if 'data' in input_yaml.keys():
        descriptor_type = input_yaml['data']['type']
    else:
        descriptor_type = 'symmetry_function'
    params_type = input_yaml['params']

    inputs = default_inputs

    for key in list(params_type.keys()):
        inputs['params'][key] = None

    data_default_inputs = get_data_default_inputs(logfile, descriptor_type=descriptor_type)
    inputs = _deep_update(inputs, data_default_inputs)
    inputs = _deep_update(inputs, preprocess_default_inputs)
    inputs = _deep_update(inputs, model_default_inputs)
    # update inputs using 'input.yaml'
    inputs = _deep_update(inputs, input_yaml, warn_new_key=True, logfile=logfile)
    #Change .T. , t to boolean
    _to_boolean(inputs)
    #add atom_types information
    if 'atom_types' not in inputs.keys():  
        inputs['atom_types'] = list(params_type.keys())
    elif not set(inputs['atom_types']) == set(params_type.keys()):
        inputs['atom_types'] = list(params_type.keys())
        logfile.write("Warning: atom_types not met with params type. Overwritting to atom_types.\n")
    else:
        logfile.write("Warning: atom_types is depreciated. Use params only.\n")


    if len(inputs['atom_types']) == 0:
        raise KeyError
    if not inputs['neural_network']['use_force'] and isinstance(inputs['preprocessing']['calc_atomic_weights'], dict):
        if inputs['preprocessing']['calc_atomic_weights']['type'] is not None:
            logfile.write("Warning: Force training is off but atomic weights are given. Atomic weights will be ignored.\n")
    if inputs['neural_network']['optimizer']['method'] == 'L-BFGS' and \
            not inputs['neural_network']['full_batch']:
        logfile.write("Warning: Optimization method is L-BFGS but full batch mode is off. This might results bad convergence or divergence.\n")

    if inputs['random_seed'] is None:
        inputs["random_seed"] = int(time.time()) 

    inputs['neural_network']['energy_coeff'] = float(inputs['neural_network']['energy_coeff'])
    inputs['neural_network']['force_coeff']  = float(inputs['neural_network']['force_coeff'])
    inputs['neural_network']['stress_coeff'] = float(inputs['neural_network']['stress_coeff'])

    if inputs['neural_network']['add_NNP_ref'] or inputs['neural_network']['train_atomic_E']:
        inputs['neural_network']['use_force'] = False
        inputs['neural_network']['use_stress'] = False

    return inputs

def get_data_default_inputs(logfile, descriptor_type='symmetry_function'):
    descriptor_inputs = {
        'symmetry_function': symmetry_function_data_default_inputs
    }

    if descriptor_type not in descriptor_inputs.keys():
        err = "'{}' type descriptor is not implemented.".format(descriptor_type)
        raise NotImplementedError(err)

    return descriptor_inputs[descriptor_type]

def _deep_update(source, overrides, warn_new_key=False, logfile=None, depth=0, parent='top'):
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

def check_inputs(inputs, logfile):
    errno = 0
    err = None
    atom_types = inputs['atom_types']
    logfile.write("{}\n".format('-'*88))
    logfile.write('\nInput for parameters\n')
    params = inputs['params']
    assert set(atom_types) == set(params.keys()), \
        f"Atom_types are not consistent with params : \
        {set(atom_types).symmetric_difference(params.keys())} "
    for atype in atom_types:
        if not os.path.exists(params[atype]):
            errno = 1
            err = f"In params {params[atype]:2} file not exist for {atype}"
        else:
            logfile.write(f"{atype:2} parameters directory     : {params[atype]}\n")
    if inputs['generate_features']:
        logfile.write("\nInput for data\n")
        data = inputs['data']
        logfile.write(f"Descriptor type             : {data['type']}\n")
        logfile.write(f"Reference data format       : {data['refdata_format']}\n")
        if data['refdata_format'] == 'vasp-out':
            logfile.write(f"Compress outcar             : {data['compress_outcar']}\n")
        logfile.write(f"Structure list              : {data['struct_list']}\n")
        logfile.write(f"Save directory              : {data['save_directory']}\n")
        logfile.write(f"Save output list            : {data['save_list']}\n")
        logfile.write(f"Use absolute path           : {data['absolute_path']}\n")
        logfile.write(f"Read force from data        : {data['read_force']}\n")
        logfile.write(f"Read stress from data       : {data['read_stress']}\n")
        logfile.write(f"Save dx as sparse tensor    : {data['dx_save_sparse']}\n")
        logfile.flush()
    if inputs['preprocess']:
        preprocessing = inputs['preprocessing']
        logfile.write('\nInput for preprocessing\n')
        logfile.write(f"Total data list             : {preprocessing['data_list']}\n")
        logfile.write(f"Train data list             : {preprocessing['train_list']}\n")
        logfile.write(f"Valid data list             : {preprocessing['valid_list']}\n")
        logfile.write(f"Valid rate                  : {preprocessing['valid_rate']}\n")
        logfile.write(f"Shuffle train/valid list    : {preprocessing['shuffle']}\n")
        logfile.write(f"Calculate scale factor      : {preprocessing['calc_scale']}\n")
        if preprocessing['calc_scale']:
            logfile.write(f"Scale type                  : {preprocessing['scale_type']}\n")
            logfile.write(f"Scale width                 : {preprocessing['scale_width']}\n")
            if preprocessing['scale_type'] == 'uniform_gas':
                for key, value in preprocessing['scale_rho'].items():
                    logfile.write(f"Scale rho for {key:2}            : {value}\n")
        logfile.write(f"Calculate PCA matrix        : {preprocessing['calc_pca']}\n")
        if preprocessing['calc_pca']:
            logfile.write(f"Use PCA whitening           : {preprocessing['pca_whiten']}\n")
            if preprocessing['pca_whiten']:
                logfile.write(f"PCA min whitening level     : {preprocessing['min_whiten_level']}\n")
        if preprocessing['calc_atomic_weights']:
            logfile.write(f"Calculate atomic_weights    : True\n")
            if preprocessing['calc_atomic_weights']['type'] in ['gdf', 'user']:
                logfile.write(f"Atomic_weights type         : {preprocessing['calc_atomic_weights']['type']}\n")
                if preprocessing['calc_atomic_weights']['params']:
                    if isinstance(preprocessing['calc_atomic_weights']['params'], dict):
                        for key, value in preprocessing['calc_atomic_weights']['params'].items():
                            logfile.write(f"sigma for {key:2}                : {value}\n")
                    else:
                        logfile.write(f"params                      : {preprocessing['calc_atomic_weights']['params']}\n")
            else:
                logfile.write("Warning : set atomic weight types appropriately. preprocessing.atomic_weights.type: gdf/user\n")
        else:
            logfile.write(f"Calculate atomic_weights    : False\n")
        logfile.flush()
    if inputs['train_model']:
        neural_network = inputs['neural_network']
        logfile.write('\nInput for neural_network\n')
        logfile.write('\nINPUT DATA\n')
        assert neural_network['train'] is False or neural_network['test'] is False, f"Invalid mode train: True, test: True. Check your input"
        logfile.write(f"Train                       : {neural_network['train']}\n")
        if neural_network['train']:
            logfile.write(f"Train list                  : {neural_network['train_list']}\n")
            logfile.write(f"Valid list                  : {neural_network['valid_list']}\n")
        logfile.write(f"Test                        : {neural_network['test']}\n")
        if neural_network['test']:
            logfile.write(f"Test_list                   : {neural_network['test_list']}\n")
        logfile.write(f"Add NNP reference to files  : {neural_network['add_NNP_ref']}\n")
        if inputs['neural_network']['add_NNP_ref'] is True:
            logfile.write(f"Reference list              : {neural_network['ref_list']}\n")
        logfile.write(f"Train atomic energy         : {neural_network['train_atomic_E']}\n")
        logfile.write(f"Use force in traning        : {neural_network['use_force']}\n")
        logfile.write(f"Use stress in training      : {neural_network['use_stress']}\n")
        logfile.write(f"Shuffle dataloader          : {neural_network['shuffle_dataloader']}\n")
        logfile.write("\nNETWORK\n")
        logfile.write(f"Nodes                       : {neural_network['nodes']}\n")
        for node in neural_network['nodes'].split('-'):
            assert node.isdigit(), f"In valid node information nodes : {neural_network['nodes']}"
        logfile.write(f"Activation function type    : {neural_network['acti_func']}\n")
        logfile.write(f"Double precision            : {neural_network['double_precision']}\n")
        logfile.write(f"Dropout ratio               : {neural_network['dropout']}\n")
        logfile.write(f"Weight initializer          : {neural_network['weight_initializer']['type']}\n")
        use_param = False
        for keys in neural_network['weight_initializer']['params'].keys():
            if neural_network['weight_initializer']['params'][keys]:
                logfile.write(f"params.{keys:4}                 : {neural_network['weight_initializer']['params'][keys]}\n")
                use_param = True
        logfile.write(f"Use scale                   : {neural_network['use_scale']}\n")
        logfile.write(f"Use PCA                     : {neural_network['use_pca']}\n")
        logfile.write(f"Use atomic_weights          : {neural_network['use_atomic_weights']}\n")
        if neural_network['use_atomic_weights']:
            if neural_network['weight_modifier']['type'] == 'modified sigmoid':
                logfile.write(f"Weight modifier type        : {neural_network['weight_modifier']['type']}\n")
                if neural_network['weight_modifier']['params']:
                    logfile.write(f" ---Parameters for weight modifier--- \n")
                    for atype in neural_network['weight_modifier']['params'].keys():
                        logfile.write(f"{atype:2}  params  : ")
                        for key, val in neural_network['weight_modifier']['params'][atype].items():
                            logfile.write(f" ({key} = {val}) ")
                        logfile.write("\n")
            elif neural_network['weight_modifier']['type']:
                logfile.write("Warning: We only support 'modified sigmoid'\n")
                logfile.write(f"Weight modifier type        : None\n")
            else:
                logfile.write(f"Weight modifier type        : None\n")
 
        logfile.write("\nOPTIMIZATION\n")
        logfile.write(f"Optimization method         : {neural_network['optimizer']['method']}\n")
        if neural_network['optimizer']['params']:
            logfile.write(f"  ---Optimizer parameters--- \n")
            for key, val in neural_network['optimizer']['params'].items():
                logfile.write(f"{key} : {val} \n")
            logfile.write('\n')
        if not neural_network['full_batch']:
            logfile.write(f"Batch size                  : {neural_network['batch_size']}\n")
        logfile.write(f"Use full batch              : {neural_network['full_batch']}\n")
        logfile.write(f"Total traning epoch         : {neural_network['total_epoch']}\n")
        logfile.write(f"Learning rate               : {neural_network['learning_rate']}\n")
        if neural_network['decay_rate']:
            logfile.write(f"Learning rate decay (exp)   : {neural_network['decay_rate']}\n")
        if neural_network['l2_regularization']:
            logfile.write(f"L2_regularization           : {neural_network['l2_regularization']}\n")
        logfile.write("\nLOSS FUNCTION\n")
        logfile.write(f"Scale for loss function     : {neural_network['loss_scale']}\n")
        logfile.write(f"Energy loss function type   : {neural_network['E_loss_type']}\n")
        logfile.write(f"Force loss function type    : {neural_network['F_loss_type']}\n")
        logfile.write(f"Energy coefficient          : {neural_network['energy_coeff']}\n")
        if neural_network['use_force']:
            logfile.write(f"Force coefficient           : {neural_network['force_coeff']}\n")
        if neural_network['use_stress']:
            logfile.write(f"Stress coefficient          : {neural_network['stress_coeff']}\n")
        logfile.write("\nLOGGING & SAVING\n")
        logfile.write(f"Show interval               : {neural_network['show_interval']}\n")
        if neural_network['save_interval'] > 0:
            logfile.write(f"Save interval               : {neural_network['save_interval']}\n")
        if neural_network['energy_criteria'] is not None:
            assert float(neural_network['energy_criteria']) > 0, f"Invalid value for energy_criteria : {neural_network['energy_criteria']}"
            logfile.write(f"Stop criteria for energy    : {neural_network['energy_criteria']}\n")
        if neural_network['use_force'] and neural_network['force_criteria'] is not None:
            assert float(neural_network['force_criteria']) > 0, f"Invalid value for force_criteria : {neural_network['force_criteria']}"
            logfile.write(f"Stop criteria for force     : {neural_network['force_criteria']}\n")
        if neural_network['use_stress'] and neural_network['stress_criteria'] is not None:
            assert float(neural_network['stress_criteria']) > 0, f"Invalid value for stress_criteria : {neural_network['stress_criteria']}"
            logfile.write(f"Stop criteria for stress    : {neural_network['stress_criteria']}\n")
        logfile.write(f"Print structure RMSE        : {neural_network['print_structure_rmse']}\n")
        #logfile.write(f"stop traning criterion if T < V : {neural_network['break_man']}\n")
        if neural_network['continue']:
            logfile.write("\nCONTINUE\n")
            if neural_network['continue'] == 'weights':
                logfile.write(f"Read neural network model parameters from ./potential_saved\n")
            else:
                logfile.write(f"Continue from checkpoint    : {neural_network['continue']}\n")
            logfile.write(f"Clear previous epoch        : {neural_network['clear_prev_status']}\n")
            logfile.write(f"Clear previous optimizer    : {neural_network['clear_prev_optimizer']}\n")
            if not neural_network["clear_prev_status"]:
                logfile.write(f"Start epoch                 : {neural_network['start_epoch']}\n")
        logfile.write("\nPARALLELISM\n")
        logfile.write(f"GPU training                : {neural_network['use_gpu']}\n")
        if inputs['neural_network']['intra_op_threads'] != 0:
            torch.set_num_threads(inputs['neural_network']['intra_op_threads'])
        if inputs['neural_network']['inter_op_threads'] != 0:
            torch.set_num_interop_threads(inputs['neural_network']['inter_op_threads'])
        logfile.write(f"Intra op parallelism thread : {torch.get_num_threads()}\n")
        logfile.write(f"Inter op parallelism thread : {torch.get_num_interop_threads()}\n")
        if neural_network['GPU_number'] is not None and torch.cuda.is_available():
            logfile.write(f"Use GPU device number       : {neural_network['GPU_number']}\n")
            assert neural_network['GPU_number'] <= torch.cuda.device_count()-1,\
             f"Invalid GPU device number available GPU # {torch.cuda.device_count()-1} , set number {neural_network['GPU_number']} "
        logfile.write(f"# of subprocesses in loading: {neural_network['subprocesses']}\n")
        logfile.flush()
    logfile.write("\n{}\n".format('-'*88))
    logfile.flush()
    return errno, err

def _to_boolean(inputs):
    check_list =  ['generate_features', 'preprocess',  'train_model']
    data_list = ['compress_outcar','read_force','read_stress', 'dx_save_sparse', 'absolute_path']
    preprocessing_list = ['shuffle', 'calc_pca', 'pca_whiten', 'calc_scale']
    neural_network_list = ['train', 'test', 'add_NNP_ref', 'train_atomic_E', 'shuffle_dataloader', 'double_precision', 'use_force', 'use_stress',\
                        'full_batch', 'print_structure_rmse', 'accurate_train_rmse', 'use_pca', 'use_scale', 'use_atomic_weights',\
                        'clear_prev_status', 'clear_prev_optimizer', 'use_gpu']

    #True TRUE T tatrue TrUe .T. ... 
    #False FALSE F f false FaLse .F. ... 
    def convert(dic, dic_key):
        check = dic[dic_key].upper()
        if check[0] == '.':
            if check[1] == 'T':
                check = True
            elif check[1] == 'F':
                check = False
            else:
                pass
        elif check[0] == 'T':
            check = True
        elif check[0] == 'F':
            check = False
        else:
            pass
        dic[dic_key] = check

    for key in check_list:
        if not isinstance(inputs[key], bool) and isinstance(inputs[key], str):
            convert(inputs, key)

    for d_key in data_list:
        if not isinstance(inputs['data'][d_key], bool) and isinstance(inputs['data'][d_key], str):
            convert(inputs['data'], d_key)

    for p_key in preprocessing_list:
        if not isinstance(inputs['preprocessing'][p_key], bool) and isinstance(inputs['preprocessing'][p_key], str):
            convert(inputs['preprocessing'], p_key)

    for n_key in neural_network_list:
        if not isinstance(inputs['neural_network'][n_key], bool) and isinstance(inputs['neural_network'][n_key], str):
            convert(inputs['neural_network'], n_key)
