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
symmetry_function_descriptor_default_inputs = \
        {'descriptor':
            {
                'type'          :   'symmetry_function',
                'struct_list'   :   './structure_list',
                'save_list'     :   './total_list',
                'save_directory':   './data',
                #data format
                'refdata_format':   'vasp-out',
                'compress_outcar':  True,
                'read_force'    :   True,
                'read_stress'   :   False,
                'dx_save_sparse':   True,
                'add_atom_idx'  :   True,
                'absolute_path' :   True,
            }
        }
preprocess_default_inputs = \
        {'preprocessing':
            {
                'data_list' : './total_list',
                'train_list': './train_list',
                'valid_list': './valid_list',
                'shuffle'   : True,
                'valid_rate': 0.1,
                #PCA
                'calc_pca'  : True,
                'pca_whiten': True,
                'pca_min_whiten_level': 1e-8,
                #Scale
                'calc_scale': True,
                'scale_type': 'minmax',
                'scale_width': 1.0,
                'scale_rho' : None,
                #GDF
                'calc_atomic_weights': {
                    'type'  : None,
                    'params': dict(),
                },
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
                'add_NNP_ref'   : False,
                'train_atomic_E': False,

                'workers'   : 0,
                'shuffle_dataloader': True,
                'double_precision'  : True,

                # Network related
                'nodes'     : '30-30',
                'use_force' : True,
                'use_stress': False,

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
                'acti_func' : 'sigmoid',
                'dropout'   : False,
                # Optimization related
                'batch_size'    : 20,
                'full_batch'    : False,
                'total_epoch'   : 1000,
                'learning_rate' : 0.0001,
                'energy_coeff'  : 1.,
                'force_coeff'   : 0.1,
                'stress_coeff'  : 0.000001,
                'loss_scale'    : 1.,
                'optimizer' : {
                    'method': 'Adam',
                    'params':
                        None
                },
                'lr_decay'      : None,
                'regularization': 1e-6, #L2 regularization
                # Loss function related
                'E_loss_type'   : 0,
                'F_loss_type'   : 1,
                # Logging & saving related (Epoch)
                'show_interval' : 10,
                'checkpoint_interval':  False,
                'energy_criteria'   :   None,
                'force_criteria'    :   None,
                'stress_criteria'   :   None,
                'break_max'         :   10,
                'print_structure_rmse': False,
                'accurate_train_rmse':  True,
                #Using preprocessing results
                'pca'   : True,
                'scale' : True,
                'atomic_weights'   : False,
                #GDF weight modifier
                'weight_modifier': {
                    'type'  : None,
                    'params': dict(),
                },
                #RESUME parameters
                'continue'      : None, # checkpoint file name  or 'weights'
                'start_epoch'   : 1,
                'clear_prev_status'     : False,
                'clear_prev_optimizer'  : False,
                #Parallelism
                'inter_op_parallelism_threads': 0,
                'intra_op_parallelism_threads': 0,
                'load_data_to_gpu'  : False,
                'GPU_number'       : None
            }
        }

def initialize_inputs(input_file_name, logfile):
    with open(input_file_name) as input_file:
        input_yaml = yaml.safe_load(input_file)
    if 'descriptor' in input_yaml.keys():
        descriptor_type = input_yaml['descriptor']['type']
    else:
        descriptor_type = 'symmetry_function'
    params_type = input_yaml['params']

    inputs = default_inputs

    for key in list(params_type.keys()):
        inputs['params'][key] = None

    descriptor_default_inputs = get_descriptor_default_inputs(logfile, descriptor_type=descriptor_type)
    inputs = _deep_update(inputs, descriptor_default_inputs)
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
    inputs['neural_network']['force_coeff'] = float(inputs['neural_network']['force_coeff'])
    inputs['neural_network']['stress_coeff'] = float(inputs['neural_network']['stress_coeff'])


    return inputs

def get_descriptor_default_inputs(logfile, descriptor_type='symmetry_function'):
    descriptor_inputs = {
        'symmetry_function': symmetry_function_descriptor_default_inputs
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

def check_inputs(inputs, logfile, run_type, error=False):
    errno = 0
    err = None
    atom_types = inputs['atom_types']
    #Check input valid and write log
    logfile.write("\n----------------------------------------------------------------------------------------\n")
    if run_type  == 'generate':
        logfile.write("\nInput for descriptor\n")
        descriptor = inputs['descriptor']
        logfile.write(f"Descriptor type             : {descriptor['type']}\n")
        params = inputs['params']
        if error: assert set(atom_types)  == set(params.keys()), f"atom_types not consistant with params : {set(atom_types).symmetric_difference(params.keys())} "
        for atype in atom_types:
            if not os.path.exists(params[atype]):
                errno = 1
                err = f"In params {params[atype]:2} file not exist for {atype}"
            else:
                logfile.write(f"{atype:2} parameters directory     : {params[atype]}\n")
        logfile.write(f"reference data format       : {descriptor['refdata_format']}\n")
        logfile.write(f"compress outcar             : {descriptor['compress_outcar']}\n")
        if error: assert os.path.exists(descriptor['struct_list']) ,f"structure list to generate : {descriptor['struct_list']} not exists." 
        logfile.write(f"structure list              : {descriptor['struct_list']}\n")
        logfile.write(f"save directory              : {descriptor['save_directory']}\n")
        logfile.write(f"save output list            : {descriptor['save_list']}\n")
        logfile.write(f"save file absolute path     : {descriptor['absolute_path']}\n")
        logfile.write(f"read force from data        : {descriptor['read_force']}\n")
        logfile.write(f"read stress from data       : {descriptor['read_stress']}\n")
        logfile.write(f"save dx as sparse tensor    : {descriptor['dx_save_sparse']}\n")
        logfile.flush()
    #Check prerpcess input is valid and write log
    elif run_type  == 'preprocess':
        preprocessing = inputs['preprocessing']
        logfile.write('\nInput for preprocessing\n')
        if not inputs['generate_features']: #Already checked if use generate
            params = inputs['params']
            if error: assert set(atom_types)  == set(params.keys()), f"atom_types not consistant with params : \
            {set(atom_types).symmetric_difference(params.keys())} "
            for atype in atom_types:
                if not os.path.exists(params[atype]):
                    errno = 1
                    err = f"In params {params[atype]:2} file not exist for {atype}"
                else:
                    logfile.write(f"{atype:2} parameters directory     : {params[atype]}\n")
        logfile.write(f"total data list             : {preprocessing['data_list']}\n")
        if error: assert os.path.exists(preprocessing['data_list']), f"data list : {preprocessing['data_list']} not exists."
        logfile.write(f"splited train list          : {preprocessing['train_list']}\n")
        logfile.write(f"splited valid list          : {preprocessing['train_list']}\n")
        logfile.write(f"valid rate                  : {preprocessing['valid_rate']}\n")
        logfile.write(f"shuffle train/valid list    : {preprocessing['shuffle']}\n")
        logfile.write(f"calculate scale factor      : {preprocessing['calc_scale']}\n")
        if preprocessing['calc_scale']:
            logfile.write(f"scale type                  : {preprocessing['scale_type']}\n")
            logfile.write(f"scale width                 : {preprocessing['scale_width']}\n")
            if preprocessing['scale_type'] == 'uniform_gas':
                for key, value in preprocessing['scale_rho'].items():
                    logfile.write(f"scale rho for {key:2}            : {value}\n")
        logfile.write(f"calculate pca matrix        : {preprocessing['calc_pca']}\n")
        if preprocessing['calc_pca']:
            if error: assert preprocessing['calc_scale'] is not False,\
             f"calculating PCA matrix must need scale factor. use calc_factor : true or filename to load"
            logfile.write(f"use pca whitening           : {preprocessing['pca_whiten']}\n")
            if preprocessing['pca_whiten']:
                logfile.write(f"pca min whitening level     : {preprocessing['pca_min_whiten_level']}\n")
        if preprocessing['calc_atomic_weights']:
            logfile.write(f"calc_atomic_weights         : True\n")
            if preprocessing['calc_atomic_weights']['type'] in ['gdf', 'user']:
                logfile.write(f"atomic_weights type         : {preprocessing['calc_atomic_weights']['type']}\n")
                if preprocessing['calc_atomic_weights']['params']:
                    if isinstance(preprocessing['calc_atomic_weights']['params'], dict):
                        for key, value in preprocessing['calc_atomic_weights']['params'].items():
                            logfile.write(f"sigma for {key:2}                : {value}\n")
                    else:
                        logfile.write(f"params                       : {preprocessing['calc_atomic_weights']['params']}\n")
            else:
                logfile.write("Warning : set atomic weight types appropriately. preprocessing.atomic_weights.type : gdf/user\n")
        else:
            logfile.write(f"calc_atomic_weights         : {preprocessing['calc_atomic_weights']}\n")
        logfile.flush()

    #Check train model input is valid and write log
    elif run_type  == 'train_model':
        neural_network = inputs['neural_network']
        logfile.write('Input for neural_network\n\n')
        if not inputs['generate_features'] and not inputs['preprocess']: #Already checked if use generate or preprocess
            params = inputs['params']
            if error: assert set(atom_types)  == set(params.keys()), f"atom_types not consistant with params : \
            {set(atom_types).symmetric_difference(params.keys())} "
            for atype in atom_types:
                if not os.path.exists(params[atype]):
                    errno = 1
                    err = f"In params {params[atype]:2} file not exist for {atype}"
                else:
                    logfile.write(f"{atype:2} parameters directory         : {params[atype]}\n")
 
        logfile.write('\n')
        logfile.write('  INPUT DATA\n')
        logfile.write(f"train                           : {neural_network['train']}\n")
        if inputs['preprocess'] is False and neural_network['train']:
            logfile.write(f"train list                      : {neural_network['train_list']}\n")
            if error: assert os.path.exists(neural_network['train_list']), f"No train_list file for training set :{neural_network['train_list']}"
            if os.path.exists(neural_network['valid_list']):
                logfile.write(f"valid list                      : {neural_network['valid_list']}\n")
        logfile.write(f"test                            : {neural_network['test']}\n")
        if neural_network['test']:
            logfile.write(f"test_list                       : {neural_network['test_list']}\n")
        if error: assert neural_network['train'] is True or neural_network['test'] is True, f"In valid mode train : false, test : false. Check your input"
        logfile.write(f"add NNP reference to files      : {neural_network['add_NNP_ref']}\n")
        if inputs['neural_network']['add_NNP_ref'] is True:
            logfile.write(f"reference list                  : {neural_network['ref_list']}\n")

        logfile.write(f"train atomic energy             : {neural_network['train_atomic_E']}\n")
        logfile.write(f"shuffle dataloader              : {neural_network['shuffle_dataloader']}\n")
        logfile.write("\n  NETWORK\n")
        logfile.write(f"nodes                           : {neural_network['nodes']}\n")
        for node in neural_network['nodes'].split('-'):
            if error: assert node.isdigit(), f"In valid node information nodes : {neural_network['nodes']}"
        logfile.write(f"use force in traning            : {neural_network['use_force']}\n")
        logfile.write(f"use stress in training          : {neural_network['use_stress']}\n")
        logfile.write(f"double precision                : {neural_network['double_precision']}\n")
        logfile.write(f"activation function type        : {neural_network['acti_func']}\n")
        logfile.write(f"use dropout network             : {neural_network['dropout']}\n")
        logfile.write(f"weight initializer              : {neural_network['weight_initializer']['type']}\n")
        use_param = False
        for keys in neural_network['weight_initializer']['params'].keys():
            if neural_network['weight_initializer']['params'][keys]:
                logfile.write(f"params.{keys}               : {neural_network['weight_initializer']['params'][keys]}\n")
                use_param = True
        logfile.write(f"use pca in traning              : {neural_network['pca']}\n")

        if neural_network['pca'] and not neural_network['continue']:
            if type(neural_network['pca']) is not bool:
                if error: assert os.path.exists(neural_network['pca']), f"{neural_network['pca']} file not exist.. set pca = False or make pca file\n"
            else:
                if error: assert  os.path.exists('./pca'), f"./pca file not exist.. set pca = False or make pca file\n"
        logfile.write(f"use scale in traning            : {neural_network['scale']}\n")
        if neural_network['scale'] and not neural_network['continue']:
            if type(neural_network['scale']) is not bool:
                if error: assert  os.path.exists(neural_network['scale']), f"{neural_network['scale']} file not exist.. set pca = False or make pca file\n"
            else:
                if error: assert  os.path.exists('./scale_factor'), f"./scale_factor file not exist.. set scale = False or make scale_factor file\n"
        logfile.write(f"use atomic_weights in traning   : {neural_network['atomic_weights']}\n")
        if neural_network['atomic_weights']:
            if neural_network['weight_modifier']['type'] != 'modified sigmoid':
                logfile.write("Warning: We only support 'modified sigmoid'\n")
            else:
                logfile.write(f"\nWeight modifier type      : {neural_network['weight_modifier']['type']}\n")
                if neural_network['weight_modifier']['params']:
                    logfile.write(f" ---parameters for weight modifier--- \n")
                    for atype in neural_network['weight_modifier']['params'].keys():
                        logfile.write(f"{atype:2}  params  : ")
                        for key, val in neural_network['weight_modifier']['params'][atype].items():
                            logfile.write(f" ({key} = {val}) ")
                        logfile.write("\n")
 
        logfile.write("\n  OPTIMIZATION\n")
        logfile.write(f"optimization method             : {neural_network['optimizer']['method']}\n")
        if neural_network['optimizer']['params']:
            logfile.write(f"  ---optimizer parameters--- \n")
            for key, val in neural_network['optimizer']['params'].items():
                logfile.write(f"{key} : {val} \n")
            logfile.write('\n')
        if not neural_network['full_batch']:
            logfile.write(f"batch size                      : {neural_network['batch_size']}\n")
        logfile.write(f"use full batch for input        : {neural_network['full_batch']}\n")
        logfile.write(f"total traning epoch             : {neural_network['total_epoch']}\n")
        logfile.write(f"learning rate                   : {neural_network['learning_rate']}\n")
        if neural_network['lr_decay']:
            logfile.write(f"learning rate decay (exp)       : {neural_network['lr_decay']}\n")
        if neural_network['regularization']:
            logfile.write(f"regularization (L2)             : {neural_network['regularization']}\n")
        logfile.write("\n  LOSS FUNCTION\n")
        logfile.write(f"energy coefficient              : {neural_network['energy_coeff']}\n")
        if neural_network['use_force']:
            logfile.write(f"force coefficient               : {neural_network['force_coeff']}\n")
        if neural_network['use_stress']:
            logfile.write(f"stress coefficient              : {neural_network['stress_coeff']}\n")
        logfile.write(f"scale for loss function         : {neural_network['loss_scale']}\n")
        logfile.write(f"energy loss function type       : {neural_network['E_loss_type']}\n")
        logfile.write(f"force  loss function type       : {neural_network['F_loss_type']}\n")
        logfile.write("\n  LOGGING & SAVING\n")
        logfile.write(f"interval (epoch) for show       : {neural_network['show_interval']}\n")
        if neural_network['checkpoint_interval']:
            logfile.write(f"interval (epoch) for checkpoint : {neural_network['checkpoint_interval']}\n")
        if neural_network['energy_criteria'] is not None:
            if error: assert float(neural_network['energy_criteria']) > 0, f"Invalid value for energy_criteria : {neural_netowkr['energy_criteria']}"
            logfile.write(f"stop criteria for energy (RMSE) : {neural_network['energy_criteria']}\n")
        if neural_network['use_force'] and neural_network['force_criteria'] is not None:
            if error: assert float(neural_network['force_criteria']) > 0, f"Invalid value for force_criteria : {neural_netowkr['force_criteria']}"
            logfile.write(f"stop criteria for force (RMSE)  : {neural_network['force_criteria']}\n")
        if neural_network['use_stress'] and neural_network['stress_criteria'] is not None:
            if error: assert float(neural_network['stress_criteria']) > 0, f"Invalid value for stress_criteria : {neural_netowkr['stress_criteria']}"
            logfile.write(f"stop criteria for stress (RMSE) : {neural_network['stress_criteria']}\n")
        logfile.write(f"print structure RMSE            : {neural_network['print_structure_rmse']}\n")
        #logfile.write(f"stop traning criterion if T < V : {neural_network['break_man']}\n")
        if neural_network['continue']:
            logfile.write("\n  CONTINUE\n")
            logfile.write(f"continue from checkpoint     : {neural_network['continue']}\n")
            if neural_network['continue'] == 'weights':
                if error: assert os.path.exists('./potential_saved'), "neural_network.continue : weights must need LAMMPS potential. Set potential ./potential_saved" 
                logfile.write(f"read neural network model parameters from ./potential_saved\n")
            else:
                if error: assert os.path.exists(neural_network['continue']), "Cannot find checkpoint file : {neural_network['continue']}. Please set file right or neural_network.contiue : false " 
            logfile.write(f"clear previous status (epoch)   : {neural_network['clear_prev_status']}\n")
            logfile.write(f"clear previous optimizer          : {neural_network['clear_prev_optimizer']}\n")
            if not neural_network["clear_prev_status"]:
                logfile.write(f"start epoch         : {neural_network['start_epoch']}\n")
        logfile.write("\n  PARALLELISM\n")
        logfile.write(f"load data directly to gpu       : {neural_network['load_data_to_gpu']}\n")
        logfile.write(f"number of workers in dataloader : {neural_network['workers']}\n")
        if neural_network['load_data_to_gpu']:
            if error: assert neural_network['workers'] == 0, f"If load data to gpu directly, use workers = 0"
        logfile.write(f"CPU core number in pytorch      : {neural_network['intra_op_parallelism_threads']}\n")
        logfile.write(f"Thread number in pytorch        : {neural_network['inter_op_parallelism_threads']}\n")
        if neural_network['GPU_number'] is not None and torch.cuda.is_available():
            logfile.write(f"Use GPU device number           : {neural_network['GPU_number']}\n")
            if error: assert neural_network['GPU_number'] <= torch.cuda.device_count()-1,\
             f"Invalid GPU device number available GPU # {torch.cuda.device_count()-1} , set number {neural_network['GPU_number']} "
        logfile.flush()
    logfile.write('\n----------------------------------------------------------------------------------------\n')
    logfile.flush()
    return errno, err

def _to_boolean(inputs):
    check_list =  ['generate_features', 'preprocess',  'train_model']
    descriptor_list = ['compress_outcar','read_force','read_stress', 'dx_save_sparse', 'add_atom_idx', 'absolute_path']
    preprocessing_list = ['shuffle', 'calc_pca', 'pca_whiten', 'calc_scale']

    neural_network_list = ['train', 'test', 'add_NNP_ref', 'train_atomic_E', 'shuffle_dataloader', 'double_precision', 'use_force', 'use_stress',\
                        'dropout','full_batch', 'checkpoint_interval', 'print_structure_rmse', 'accurate_train_rmse', 'pca', 'scale', 'atomic_weights',\
                        'clear_prev_status', 'clear_prev_optimizer', 'load_data_to_gpu']


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

    for d_key in descriptor_list:
        if not isinstance(inputs['descriptor'][d_key], bool) and isinstance(inputs['descriptor'][d_key], str):
            convert(inputs['descriptor'], d_key)

    for p_key in preprocessing_list:
        if not isinstance(inputs['preprocessing'][p_key], bool) and isinstance(inputs['preprocessing'][p_key], str):
            convert(inputs['preprocessing'], p_key)

    for n_key in neural_network_list:
        if not isinstance(inputs['neural_network'][n_key], bool) and isinstance(inputs['neural_network'][n_key], str):
            convert(inputs['neural_network'], n_key)
