import torch
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import  Linear, Parameter

import time
import numpy as np

from simple_nn_v2.models.neural_network import FCNDict, FCN, read_lammps_potential
from simple_nn_v2.models.data_handler import StructlistDataset, FilelistDataset, _set_struct_dict, _make_dataloader, filename_collate
from simple_nn_v2.models.train import train, save_checkpoint, _show_structure_rmse, _save_nnp_result, _save_atomic_E


def train_NN(inputs, logfile, user_optimizer=None):
    if inputs['neural_network']['double_precision']: 
        torch.set_default_dtype(torch.float64)

    device = _set_device()
    model = _initialize_model(inputs, logfile, device)
    optimizer = _initialize_optimizer(inputs, model)
    criterion = torch.nn.MSELoss(reduction='none').to(device=device)    
    
    # Resume job if possible and load scale_factor, pca
    checkpoint, best_loss = _load_checkpoint(inputs, logfile, model, optimizer)
    scale_factor, pca = _load_scale_factor_and_pca(inputs, logfile, checkpoint)

    # Prepare data set
    train_dataset_list, valid_dataset_list = _load_dataset_list(inputs, logfile)
    if inputs['neural_network']['full_batch']:
        batch_size = len(train_dataset_list)
    else:
        batch_size = inputs['neural_network']['batch_size']
    train_loader, valid_loader = _make_dataloader(inputs, logfile, scale_factor, pca, train_dataset_list, valid_dataset_list, batch_size=batch_size)   

    # For structure rmse    
    train_struct_dict, valid_struct_dict = _load_structure(inputs, logfile, scale_factor, pca)
    
    # Run training
    _do_train(inputs, logfile, train_loader, valid_loader, model, optimizer, criterion, scale_factor, pca, best_loss, train_struct_dict, valid_struct_dict) 

def _initialize_model(inputs, logfile, device):
    logfile.write(f"Use {device} in model\n")
    #Parrelism
    if inputs['neural_network']['intra_op_parallelism_threads'] != 0:
        torch.set_num_threads(inputs['neural_network']['intra_op_parallelism_threads'])
    if inputs['neural_network']['inter_op_parallelism_threads'] != 0:
        torch.set_num_interop_threads(inputs['neural_network']['inter_op_parallelism_threads'])
    logfile.write("Parallelism intra_thread : {0} inter_thread : {1}\n".format(torch.get_num_threads(),torch.get_num_interop_threads()))

    #Set default configuration
    model = {}
    for item in inputs['atom_types']:
        temp_nodes = [int(jtem) for jtem in inputs['neural_network']['nodes'].split('-')]
        #Extract number of symmetry functions in params
        with open(inputs['descriptor']['params'][item], 'r') as f:
            tmp_symf = f.readlines()
            sym_num = len(tmp_symf)
        #Make NN for each element 
        model[item] = FCN(sym_num, temp_nodes,
        acti_func = inputs['neural_network']['acti_func'],
        dropout = inputs['neural_network']['dropout']) #Make nodes per elements

        #Apply weight initialization !
        weight_log = _init_weight(inputs, logfile, model[item])
    
    logfile.write(weight_log)
    model = FCNDict(model) #Make full model with elementized dictionary model
    model.to(device=device)
    logfile.write("Initialize pytorch model\n")

    return model

def _load_checkpoint(inputs, logfile, model, optimizer):
    device = _set_device()
    loss = float('inf')
    checkpoint = None

    # Load model, optimizer, status info
    if inputs['neural_network']['continue'] == 'weights':
        _load_lammps_potential(inputs, logfile, model)
    elif inputs['neural_network']['continue']:
        checkpoint = torch.load(inputs['neural_network']['continue'])
        model.load_state_dict(checkpoint['model'])
        logfile.write("Load pytorch model from {0}\n".format(inputs['neural_network']['continue']))

        if not inputs['neural_network']['clear_prev_network']:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg in optimizer.param_groups:
                pg['lr'] = inputs['neural_network']['learning_rate']
            logfile.write("Load previous model, optimizer, learning rate\n")

        if not inputs['neural_network']['clear_prev_status']:
            loss = checkpoint['loss']
            inputs['neural_network']['start_epoch'] = checkpoint['epoch']
            logfile.write("Load previous loss : {0:6.2e}, epoch : {1}\n".format(checkpoint['loss'], checkpoint['epoch']))

    return checkpoint, loss

def _load_scale_factor_and_pca(inputs, logfile, checkpoint):
    device = _set_device()
    scale_factor = None
    pca = None

    if checkpoint is not None:
        if inputs['descriptor']['calc_scale']:
            scale_factor = checkpoint['scale_factor']
            logfile.write("Load scale factor from checkpoint\n")
        if inputs['neural_network']['pca']:
            pca = checkpoint['pca']
            logfile.write("Load pca from checkpoint\n")
    else:
        if inputs['descriptor']['calc_scale']:
            scale_factor = torch.load('./scale_factor')
            logfile.write("Scale factor data loaded\n")
        if inputs['neural_network']['pca']:
            pca = torch.load('./pca')
            logfile.write("PCA data loaded\n")        
        _convert_to_tensor(inputs, logfile, scale_factor, pca)
    
    return scale_factor, pca

def _load_lammps_potential(inputs, logfile, model):
    potential_params = read_lammps_potential('./potential_saved')
    assert [item for item in model.keys].sort() == [item for item in potential_params.keys()].sort()

    for item in potential_params.keys():
        for name, lin in model.nets[item].lin.named_modules():
            if name in potential_params[item].keys():
                lin.weight = Parameter(torch.transpose(torch.tensor(potential_params[item][name]['weight']), -1,0))
                lin.bias = Parameter(torch.transpose(torch.tensor(potential_params[item][name]['bias']), -1,0))
    logfile.write("Load parameters from lammps potential filename: {0}\n".format(inputs['neural_network']['continue']))
 
# Convert generated scale_factor, pca to pytorch tensor format 
def _convert_to_tensor(inputs, logfile, scale_factor, pca):
    device = _set_device()
    device = torch.device('cpu')
    for item in inputs['atom_types']:
        if scale_factor:
            max_plus_min  = torch.tensor(scale_factor[item][0,:], device=device)
            max_minus_min = torch.tensor(scale_factor[item][1,:], device=device)
            scale_factor[item] = [max_plus_min, max_minus_min] #To list format
            logfile.write("Convert {0} scale_factor to tensor\n".format(item))
        if pca:
            pca[item][0] = torch.tensor(pca[item][0], device=device)
            pca[item][1] = torch.tensor(pca[item][1], device=device)
            pca[item][2] = torch.tensor(pca[item][2], device=device)
            logfile.write("Convert {0} PCA to tensor\n".format(item))

def _load_dataset_list(inputs, logfile):
    device = _set_device()
    if inputs['neural_network']['test'] is False:
        train_dataset_list = FilelistDataset(inputs['descriptor']['train_list'])
        valid_dataset_list = FilelistDataset(inputs['descriptor']['valid_list']) 
        try: #Check valid dataset exist
            valid_dataset_list[0] 
            logfile.write("Train & Valid dataset loaded\n")
        except: #No validation
            valid_dataset_list = None
            logfile.write("Train dataset loaded, No valid set loaded\n")
    else:
        train_dataset_list = FilelistDataset(inputs['descriptor']['test_list'])
        valid_dataset_list = None
        logfile.write("Test dataset loaded\n")

    return train_dataset_list, valid_dataset_list
    
#Load structure dictrionary for RMSE
def _load_structure(inputs, logfile, scale_factor, pca):
    device = _set_device()
    train_struct_dict = None
    valid_struct_dict = None
    train_struct_dict = _set_struct_dict(inputs['descriptor']['train_list'])

    #Check valid list exist and test scheme
    if not inputs['neural_network']['test']:
        valid_struct_dict = _set_struct_dict(inputs['descriptor']['valid_list']) 
        #Dictionary Key merge (in train structure, not in valid structue)
        for t_key in train_struct_dict.keys():
            if not t_key in valid_struct_dict.keys():
                #Set blank dataframe
                valid_struct_dict[t_key] = StructlistDataset()
    else:
        valid_struct_dict = dict()
        for t_key in train_struct_dict.keys():
            #Set blank dataframe
            valid_struct_dict[t_key] = StructlistDataset()

    #Loop for _make_dataloader
    for t_key in train_struct_dict.keys():
        if inputs['neural_network']['full_batch']:
            batch_size = len(train_struct_dict[t_key])
        else:
            batch_size = inputs['neural_network']['batch_size']
        train_struct_dict[t_key], valid_struct_dict[t_key] = _make_dataloader(inputs,
         logfile, scale_factor, pca, train_struct_dict[t_key], valid_struct_dict[t_key], batch_size=batch_size)

    return train_struct_dict, valid_struct_dict

#Main traning part 
def _do_train(inputs, logfile, train_loader, valid_loader, model, optimizer, criterion, scale_factor, pca, best_loss, train_struct_dict=None, valid_struct_dict=None):
    device = _set_device()
    start_time = time.time()

    #Calculate total epoch, batch size
    max_len = len(train_loader)
    total_epoch = int(inputs['neural_network']['total_epoch'])
    total_iter = int(inputs['neural_network']['total_epoch']*max_len)
    logfile.write("Total training iteration : {0} , epoch : {1}, batch number : {2}, batch size : {3}, workers : {4}\n"\
    .format(total_iter, total_epoch, max_len,'full_batch' if inputs['neural_network']['full_batch'] else inputs['neural_network']['batch_size'],inputs['neural_network']['workers']))
    
    #Learning rate decay schedular
    if inputs['neural_network']['lr_decay']:
        scheduler = ExponentialLR(optimizer=optimizer, gamma=inputs['neural_network']['lr_decay'])
    
    #Check use validation (valid set exist)
    if valid_loader:
        valid = True
    else:
        valid = False
        
    #Define energy, force, stress error dictionary to use stop criteria
    err_dict = _check_criteria(inputs, logfile)
    
    if inputs['neural_network']['save_result'] or inputs['descriptor']['add_NNP_ref']:
        train_dataset_save = FilelistDataset(inputs['descriptor']['train_list'])
        valid_dataset_save = FilelistDataset(inputs['descriptor']['valid_list'])
        if inputs['descriptor']['add_NNP_ref']:
            train_dataset_save.save_filename()
            valid_dataset_save.save_filename()
            trainset_saved, validset_saved = _make_dataloader(inputs, logfile, scale_factor, pca,
            train_dataset_save, valid_dataset_save, batch_size=inputs['neural_network']['batch_size'], my_collate=filename_collate)
        else:
            trainset_saved, validset_saved = _make_dataloader(inputs, logfile, scale_factor, pca,
            train_dataset_save, valid_dataset_save, batch_size=inputs['neural_network']['batch_size'])

    if inputs['neural_network']['test']: #Evalutaion model
        print("Evaluation(Testing) model")
        logfile.write("Evaluation(Testing) model \n")
        loss = train(inputs, logfile, train_loader, model, criterion=criterion, start_time=start_time, test=True)
        if inputs['neural_network']['print_structure_rmse'] and train_struct_dict: 
            _show_structure_rmse(inputs, logfile, train_struct_dict, valid_struct_dict, model, optimizer=optimizer, criterion=criterion, test=True)
    else: #Traning model 
        best_epoch = inputs['neural_network']['start_epoch'] #Set default value
        for epoch in range(inputs['neural_network']['start_epoch'], total_epoch+1):
            #Train  model with train loader 
            t_loss = train(inputs, logfile, train_loader, model, optimizer=optimizer, criterion=criterion, epoch=epoch, err_dict=err_dict, start_time=start_time)
            #Calculate valid loss with valid loader if valid dataset exists
            if valid: 
                loss  = train(inputs, logfile, valid_loader, model, criterion=criterion, valid=True, epoch=epoch, err_dict=err_dict, start_time=start_time)
            #No valid set, get training loss as valid loss
            else: 
                loss = t_loss 
            
            #Structure rmse part
            if (epoch % inputs['neural_network']['show_interval'] == 0) and\
                 inputs['neural_network']['print_structure_rmse'] and train_struct_dict: 
                _show_structure_rmse(inputs, logfile, train_struct_dict, valid_struct_dict, model, optimizer=optimizer, criterion=criterion)

            #Learning rate decay part
            if inputs['neural_network']['lr_decay']: scheduler.step()

            #Check best loss, epoch
            is_best = loss < best_loss
            if is_best:
                best_loss = loss
                best_epoch = epoch 
                #Checkpoint in model traning when best model
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='bestmodel.pth.tar' )

            #Checkpoint for save iteration
            if inputs['neural_network']['checkpoint_interval'] and (epoch % inputs['neural_network']['checkpoint_interval'] == 0):
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename=f'epoch_{epoch}.pth.tar')
            elif not inputs['neural_network']['checkpoint_interval']:
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor)

            #Save result
            if inputs['neural_network']['save_result'] and  (epoch % inputs['neural_network']['save_interval'] == 0):
                res_dict = _save_nnp_result(inputs, model, trainset_saved, validset_saved)
                logfile.write(f"DFT, NNP result saved at {epoch}\n")
                torch.save(res_dict, 'saved_result')

            #LAMMPS potential save part
            breaksignal  = _save_lammps(inputs, logfile, model, is_best, epoch, scale_factor, pca, err_dict)            
            if breaksignal:  
                logfile.write("Break point reached. Terminating traning model\n")
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='breakpoint.pth.tar')
                break

        #End of traning loop : best loss potential written
        logfile.write("Best loss lammps potential written at {0} epoch\n".format(best_epoch))

    if inputs['descriptor']['add_NNP_ref'] and not inputs['neural_network']['E_loss_type'] == 3:
        _save_atomic_E(inputs, logfile, model, trainset_saved, validset_saved)

#function to save lammps with criteria or epoch
def _save_lammps(inputs, logfile, model, is_best, epoch, scale_factor, pca, err_dict):
    #Lammps save part 
    #Save best lammps potential if set save_best
    if is_best: 
        model.write_lammps_potential(filename='./potential_saved_best_loss', inputs=inputs, scale_factor=scale_factor, pca=pca)
        
    #Save lammps potential with save_interval
    if epoch % inputs['neural_network']['save_interval'] == 0:
        model.write_lammps_potential(filename='./potential_saved_epoch_{0}'.format(epoch), inputs=inputs, scale_factor=scale_factor, pca=pca)
        print("Lammps potential written at {0} epoch\n".format(epoch))
        logfile.write("Lammps potential written at {0} epoch\n".format(epoch))


    #Break if energy, force, stress is under their criteria
    breaksignal = False
    if err_dict:
        for err_type in err_dict.keys():
            if err_dict[err_type][0] < err_dict[err_type][1]:
                breaksignal = True
            else:
                breaksignal = False
                
    if breaksignal:
        for err_type in err_dict.keys():
            print("Ctirerion met {0} : {1:4.2f} < criteria : {2:4}".format(err_type, err_dict[err_type][0], err_dict[err_type][1]))
            logfile.write("Ctirerion met {0} : {1:4.2f} < criteria : {2:4}\n".format(err_type, err_dict[err_type][0], err_dict[err_type][1]))
        model.write_lammps_potential(filename='./potential_criterion', inputs=inputs, scale_factor=scale_factor, pca=pca)
        print("Criterion lammps potential written".format(err_type, err_dict[err_type][0], err_dict[err_type][1]))
        logfile.write("Criterion lammps potential written\n".format(err_type, err_dict[err_type][0], err_dict[err_type][1]))

    return breaksignal

#Check energy, force, stress criteria exist and create dictoary for them
def _check_criteria(inputs,logfile):
    if inputs['neural_network']['energy_criteria'] or inputs['neural_network']['force_criteria'] or inputs['neural_network']['stress_criteria']:
        if inputs['neural_network']['energy_criteria']:
            #Dictionaly with list [error , criteria]
            err_dict = dict()
            err_dict['e_err'] = [float('inf') , float(inputs['neural_network']['energy_criteria'])]
            logfile.write("Energy criteria used : {0:4}  \n".format(float(inputs['neural_network']['energy_criteria'])))
        if inputs['neural_network']['force_criteria'] and inputs['neural_network']['use_force']:
            err_dict['f_err'] = [float('inf'), float(inputs['neural_network']['force_criteria'])]
            logfile.write("Force criteria used : {0:4}\n".format(float(inputs['neural_network']['force_criteria'])))
        if inputs['neural_network']['stress_criteria'] and inputs['neural_network']['use_stress']:
            err_dict['s_err'] = [float('inf'), float(inputs['neural_network']['stress_criteria'])]
            logfile.write("Stress criteria used : {0:4}\n".format(float(inputs['neural_network']['stress_criteria'])))
    else:
        err_dict = None

    return err_dict
   
def _init_weight(inputs, logfile, model):
    #Get parameter for initialization
    try:
        init_dic = inputs['neural_network']['weight_initializer']
        init_name = init_dic['type']
        init_params = init_dic['params']
        acti_fn = inputs['neural_network']['acti_func']

        #If use xavier initialization, gain should be defined
        if init_name in ['xavier uniform', 'xavier normal', 'orthogonal']:
            #Nonlinear functions that gain defined
            if acti_fn in ['sigmoid', 'tanh', 'relu', 'selu']:
                init_params['gain'] = init.calculate_gain(acti_fn)
            else:
                if init_params['gain'] is None:
                    logfile.write("gain must be defined for using '{}' initializer with '{}' activation function".format(init_name, acti_fn))
    except:
        init_dic = None

    implimented_init = ['xavier uniform', 'xavier normal', 'normal', 'constant', 'kaiming normal', 'he normal', 'kaiming uniform', 'he uniform', 'orthogonal']
    
    # weight initialize
    try:
        if init_dic is None:
            weight_log = "No weight initializer infomation in input file\n"
        elif init_name not in implimented_init:
            weight_log = f"{init_name} weight initializer infomation is not implemented\n".format()
        else:     
            weight_initializer, kwarg = _get_initializer_and_kwarg(init_name, init_params)
            for lin in model.lin:
                if lin == Linear:
                    weight_initializer(lin.weight, **kwarg)
                    weight_initializer(lin.bias, **kwarg)
            weight_log = "{} weight initializer : {}\n".format(init_name, kwarg)
    except:
        import sys
        print(sys.exc_info())
        weight_log = "During weight initialization error occured. Default Initializer used\n"
        
    return  weight_log

def _get_initializer_and_kwarg(init_name, init_params):
    weight_initializer = {
        'xavier uniform': init.xavier_uniform_, 
        'xavier normal': init.xavier_uniform_, 
        'normal': init.normal_, 
        'constant': init.constant_, 
        'kaiming normal': init.kaiming_normal_, 
        'kaiming uniform': init.kaiming_uniform_, 
        'he normal': init.kaiming_normal_,
        'he uniform': init.kaiming_uniform_, 
        'orthogonal': init.orthogonal_
    }
    kwarg = {
        'xavier uniform': {'gain': init_params['gain']}, 
        'xavier normal': {'gain': init_params['gain']}, 
        'normal': {'mean': init_params['mean'], 'std': init_params['std']}, 
        'constant': {'var': init_params['var']}, 
        'kaiming normal': {'mode': 'fan_out', 'nonlinearity': 'relu'}, 
        'kaiming uniform': {'mode': 'fan_out', 'nonlinearity': 'relu'}, 
        'he normal': {'mode': 'fan_out', 'nonlinearity': 'relu'}, 
        'he unifrom': {'mode': 'fan_out', 'nonlinearity': 'relu'}, 
        'orthogonal': {'gain': init_params['gain']}
    }
    
    return weight_initializer[init_name], kwarg[init_name]

def _initialize_optimizer(inputs, model):
    optim_type = inputs['neural_network']['method']
    lr=inputs['neural_network']['learning_rate']
    regularization = float(inputs['neural_network']['regularization'])

    optimizer = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }

    return optimizer[optim_type](model.parameters(), lr=lr, weight_decay=regularization)

def _set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


