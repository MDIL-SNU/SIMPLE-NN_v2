import torch
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import  Linear, Parameter

import time
import numpy as np
from sklearn.decomposition import PCA

from .neural_network import FCNDict, FCN, read_lammps_potential
from .data_handler import StructlistDataset, FilelistDataset, _set_struct_dict, _load_collate, filename_collate
from .train import train, save_checkpoint, _show_structure_rmse, _save_nnp_result, _save_atomic_E


def train_NN(inputs, logfile,user_optimizer=None):
    #Set default type
    if inputs['neural_network']['double_precision']: 
        torch.set_default_dtype(torch.float64)

    # Initialize model
    model, optimizer, criterion, scale_factor, pca = _init_model(inputs, logfile, user_optimizer=None)
    
    # Resume job if possible and load dataset & scale_factor, pca
    scale_factor, pca, train_dataset, valid_dataset, best_loss = _load_data(\
    inputs, logfile, model, optimizer, scale_factor, pca)

    # Load data loader
    if inputs['neural_network']['full_batch']:
        batch_size = len(train_dataset)
    else:
        batch_size = inputs['neural_network']['batch_size']

    train_loader, valid_loader = _load_collate(inputs, logfile, scale_factor, pca, train_dataset, valid_dataset, batch_size=batch_size)   

    # For structure rmse    
    train_struct_dict, valid_struct_dict = _load_structure(inputs, logfile, scale_factor, pca)
    
    # Run training
    _do_train(inputs, logfile, train_loader, valid_loader, model, optimizer, criterion, scale_factor, pca, best_loss, train_struct_dict, valid_struct_dict) 
    # End of progran & Best loss witten 


#Initialize model with input, and set default value scale, PCA
def _init_model(inputs, logfile, user_optimizer=None):
    #Set default configuration
    logfile.write('Initialize pytorch model\n')
    model = {}
    for item in inputs['atom_types']:
        temp_nodes = [int(jtem) for jtem in inputs['neural_network']['nodes'].split('-')]
        #Extract number of symmetry functions in params
        with open(inputs['descriptor']['params'][item],'r') as f:
            tmp_symf = f.readlines()
            sym_num = len(tmp_symf)
        #Make NN for each element 
        model[item] = FCN(sym_num, temp_nodes,
         acti_func=inputs['neural_network']['acti_func'],
         dropout=inputs['neural_network']['dropout']) #Make nodes per elements

        #Apply weight initialization !
        weight_log = _init_weight(inputs, model[item])
    
    logfile.write(weight_log)
    model = FCNDict(model) #Make full model with elementized dictionary model
    regularization = float(inputs['neural_network']['regularization'])

    if inputs['neural_network']['method'] == 'Adam':
        #Adam optimizer (Default)
        optimizer = torch.optim.Adam(model.parameters(), lr=inputs['neural_network']['learning_rate'],
     weight_decay=regularization)

    elif inputs['neural_network']['method'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=inputs['neural_network']['learning_rate'],
        weight_decay=regularization)
    else:
        if user_optimizer != None:
            optimizer = user_optimizer(model.parameters(), lr=inputs['neural_network']['learning_rate'],
            **self.inputs['neural_network']['optimizer']) 
        else:
            raise ValueError
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logfile.write(f"Use {device} in model\n")
    device = torch.device(device)
    model.to(device=device)
    criterion = torch.nn.MSELoss(reduction='none').to(device=device)

    scale_factor = None
    pca = None
    
    return  model, optimizer, criterion, scale_factor, pca

#Load data (pca,scale fator, dataset) , resume from checkpoint
def _load_data(inputs, logfile, model, optimizer, scale_factor, pca):
    if inputs["neural_network"]["continue"]:
        logfile.write('Load pytorch model from {0}\n'.format(inputs["neural_network"]["continue"]))
        checkpoint = torch.load(inputs["neural_network"]["continue"])

        model.load_state_dict(checkpoint['model'])
        # Load model
        if not inputs["neural_network"]['clear_prev_network']:
            logfile.write('Load previous model, optimizer, learning rate\n')
            #OPTIMIZER
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg in optimizer.param_groups:
                pg['lr'] = inputs['neural_network']['learning_rate']

        if not inputs['neural_network']['clear_prev_status']:
            logfile.write('Load previous loss : {0:6.2e}, epoch : {1}\n'.format(checkpoint['loss'], checkpoint['epoch']))
            loss = checkpoint['loss']
            inputs['neural_network']['start_epoch'] = checkpoint['epoch']
        else:
            loss = float('inf')

        # Read LAMMPS potential
        if inputs["neural_network"]['read_potential']:
            _load_lammps_potential(inputs, logfile, model)

        # Load scale file & PCA file
        if inputs['descriptor']['calc_scale']:
            logfile.write('Load scale from checkpoint\n')
            scale_factor = checkpoint['scale_factor']
        if inputs['neural_network']['pca']:
            logfile.write('Load pca from checkpoint\n')
            pca = checkpoint['pca']

    else: #Not resume, load scale_factor, pca pytorch savefile in preprocess
        #Set best loss as infinite
        loss = float('inf')
        if inputs['descriptor']['calc_scale']:
            scale_factor = torch.load('./scale_factor')
            logfile.write('Scale factor data loaded\n')

        if inputs['neural_network']['pca']:
            pca = torch.load('./pca')
            logfile.write('PCA data loaded\n')
        
        _convert_to_tensor(inputs, logfile, scale_factor, pca)

    train_dataset = None
    valid_dataset = None

    # Load train dataset 
    train_dataset = FilelistDataset(inputs['descriptor']['train_list'])
    # Load valid dataset
    if not inputs['neural_network']['test']:
        valid_dataset = FilelistDataset(inputs['descriptor']['valid_list']) 
        try: #Check valid dataset exist
            valid_dataset[0] 
            logfile.write('Train & Valid dataset loaded\n')
        except: #No validation
            valid_dataset = None
            logfile.write('Train dataset loaded, No valid set loaded\n')
    else:
        logfile.write('Test dataset loaded\n')

    #model, optimizer no need to be returned
    return scale_factor, pca, train_dataset, valid_dataset, loss 

#Main traning part 
def _do_train(inputs, logfile, train_loader, valid_loader, model, optimizer, criterion, scale_factor, pca, best_loss, train_struct_dict = None, valid_struct_dict = None):
    
    #Get start time 
    start_time = time.time()

    #Calculate total epoch, batch size
    max_len = len(train_loader)
    total_epoch = int(inputs['neural_network']['total_epoch'])
    total_iter = int(inputs['neural_network']['total_epoch']*max_len)
    logfile.write('Total training iteration : {0} , epoch : {1}, batch number : {2}\n'.format(total_iter, total_epoch, max_len))
    
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
            trainset_saved, validset_saved = _load_collate(inputs, logfile, scale_factor, pca,
            train_dataset_save, valid_dataset_save, batch_size=inputs['neural_network']['batch_size'], my_collate=filename_collate)
        else:
            trainset_saved, validset_saved = _load_collate(inputs, logfile, scale_factor, pca,
            train_dataset_save, valid_dataset_save, batch_size=inputs['neural_network']['batch_size'])

    
    #Evaluation model
    if inputs['neural_network']['test']: 
        print('Evaluation(Testing) model')
        logfile.write('Evaluation(Testing) model \n')
        loss = train(inputs, logfile, train_loader, model, criterion=criterion, start_time=start_time, test=True)
        if inputs['neural_network']['print_structure_rmse'] and train_struct_dict: 
            _show_structure_rmse(inputs, logfile, train_struct_dict, valid_struct_dict, model, optimizer=optimizer, criterion=criterion, test=True)
    #Traning model
    else: 
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
            if (epoch  % inputs['neural_network']['show_interval'] == 0) and\
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
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename = 'bestmodel.pth.tar' )

            #Checkpoint for save iteration
            if inputs['neural_network']['checkpoint_interval'] and (epoch  % inputs['neural_network']['checkpoint_interval'] == 0):
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename = f'epoch_{epoch}.pth.tar')
            elif not inputs['neural_network']['checkpoint_interval']:
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor)

            #Save result
            if inputs['neural_network']['save_result'] and  (epoch % inputs['neural_network']['save_interval'] == 0):
                res_dict = _save_nnp_result(inputs, model, trainset_saved, validset_saved)
                logfile.write(f'DFT, NNP result saved at {epoch}\n')
                torch.save(res_dict, 'saved_result')

            #LAMMPS potential save part
            breaksignal  = _save_lammps(inputs, logfile, model, is_best, epoch, scale_factor, pca, err_dict)            
            if breaksignal:  
                logfile.write('Break point reached. Terminating traning model\n')
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename = 'breakpoint.pth.tar')
                break

        #End of traning loop : best loss potential written
        logfile.write('Best loss lammps potential written at {0} epoch\n'.format(best_epoch))

    if inputs['descriptor']['add_NNP_ref']:
        _save_atomic_E(inputs, logfile, model, trainset_saved, validset_saved)

#function to save lammps with criteria or epoch
def _save_lammps(inputs, logfile, model, is_best, epoch, scale_factor, pca, err_dict):
    #Lammps save part 
    #Save best lammps potential if set save_best
    if is_best: 
        model.write_lammps_potential(filename ='./potential_saved_best_loss', inputs=inputs, scale_factor=scale_factor, pca=pca)
        
    #Save lammps potential with save_interval
    if epoch  % inputs['neural_network']['save_interval'] == 0:
        model.write_lammps_potential(filename ='./potential_saved_epoch_{0}'.format(epoch), inputs=inputs, scale_factor=scale_factor, pca=pca)
        print('Lammps potential written at {0} epoch\n'.format(epoch))
        logfile.write('Lammps potential written at {0} epoch\n'.format(epoch))


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
            print('Ctirerion met {0} : {1:4.2f} < criteria : {2:4}'.format(err_type,err_dict[err_type][0],err_dict[err_type][1]))
            logfile.write('Ctirerion met {0} : {1:4.2f} < criteria : {2:4}\n'.format(err_type,err_dict[err_type][0],err_dict[err_type][1]))
        model.write_lammps_potential(filename ='./potential_criterion', inputs=inputs, scale_factor=scale_factor, pca=pca)
        print('Criterion lammps potential written'.format(err_type,err_dict[err_type][0],err_dict[err_type][1]))
        logfile.write('Criterion lammps potential written\n'.format(err_type,err_dict[err_type][0],err_dict[err_type][1]))

    return breaksignal

#Check energy, force, stress criteria exist and create dictoary for them
def _check_criteria(inputs,logfile):
    if inputs['neural_network']['energy_criteria'] or inputs['neural_network']['force_criteria'] or inputs['neural_network']['stress_criteria']:
        if inputs['neural_network']['energy_criteria']:
            #Dictionaly with list [error , criteria]
            err_dict = dict()
            err_dict['e_err'] = [float('inf') , float(inputs['neural_network']['energy_criteria'])]
            logfile.write('Energy criteria used : {0:4}  \n'.format(float(inputs['neural_network']['energy_criteria'])))
        if inputs['neural_network']['force_criteria'] and inputs['neural_network']['use_force']:
            err_dict['f_err'] = [float('inf'), float(inputs['neural_network']['force_criteria'])]
            logfile.write('Force criteria used : {0:4}\n'.format(float(inputs['neural_network']['force_criteria'])))
        if inputs['neural_network']['stress_criteria'] and inputs['neural_network']['use_stress']:
            err_dict['s_err'] = [float('inf'), float(inputs['neural_network']['stress_criteria'])]
            logfile.write('Stress criteria used : {0:4}\n'.format(float(inputs['neural_network']['stress_criteria'])))
    else:
        err_dict = None

    return err_dict

#Load structure dictrionary for RMSE
def _load_structure(inputs, logfile, scale_factor, pca):
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

    #Loop for _load_collate
    for t_key in train_struct_dict.keys():
        if inputs['neural_network']['full_batch']:
            batch_size = len(train_struct_dict[t_key])
        else:
            batch_size = inputs['neural_network']['batch_size']
        train_struct_dict[t_key], valid_struct_dict[t_key] =_load_collate(inputs,
         logfile, scale_factor, pca, train_struct_dict[t_key], valid_struct_dict[t_key],batch_size=batch_size)

    return train_struct_dict, valid_struct_dict

#weight, bias initialization here
def _init_weight(inputs, model):
    #Get parameter for initialization
    try:

        init_dic = inputs['neural_network']['weight_initializer']
        init_name = init_dic['type']
        init_params = init_dic['params']

        #If use xavier initialization, gain should be defined
        if init_name == 'xavier uniform' or init_name == 'xavier normal' or init_name == 'orthogonal':
            anti = inputs['neural_network']['acti_func']
            #Nonlinear functions that gain defined
            nonlinear_list = ['sigmoid', 'tanh', 'relu', 'selu']
            if anti in nonlinear_list:
                gain = init.calculate_gain(anti)
            else:
                gain = init_params['gain']
    except:
        init_dic = None
    
    implimented_init = ['xavier uniform', 'xavier normal', 'normal', 'constant', 'kaiming normal', 'he normal', 
                        'kaiming uniform', 'he uniform', 'orthogonal']
    try:
        #Loop for initilaization
        if not init_dic:
            weight_log = "No weight initialization infomation in input file\n"

        elif not init_name in implimented_init:
            weight_log = f"{init_name} weight initialization infomation is not implemented\n".format()

        elif init_name == 'xavier uniform':
            weight_log = "Xavier Uniform weight initialization : gain {0:4.2f}\n".format(gain)
            for lin in model.lin:
                if lin == Linear:
                    init.xavier_uniform_(lin.weight, gain=gain)
                    init.xavier_unifrom_(lin.bias, gain=gain)

        elif init_name == 'xavier normal':
            weight_log = "Xavier Uniform weight initialization : gain {0:4.2f}\n".format(gain)
            for lin in model.lin:
                if lin == Linear:
                    init.xavier_normal_(lin.weight, gain=gain)
                    init.xavier_normal_(lin.bias, gain=gain)

        elif init_name == 'normal':
            weight_log = "Normal weight initialization : mean {0} std {1}\n".format(init_params['mean'], init_params['std'])
            for lin in model.lin:
                if lin == Linear:
                    init.normal_(lin.weight, mean=init_params['mean'], std=init_params['std'])
                    init.normal_(lin.bias, mean=init_params['mean'], std=init_params['std'])

        elif init_name == 'constant':
            weight_log = "Constant weight initialization : var {0}\n".format(init_params['var'])
            for lin in model.lin:
                if lin == Linear:
                    init.constant_(lin.weight, var=init_params['var'])
                    init.constant_(lin.bias, var=init_params['var'])

        elif init_name == 'kaiming uniform' or init_name == 'he uniform':
            weight_log = "Kaiming Uniform (He Uniform) weight initialization\n"
            for lin in model.lin:
                if lin == Linear:
                    init.kaiming_uniform_(lin.weight, mode='fan_out', nonlinearity='relu')
                    init.kaiming_uniform_(lin.bias, mode='fan_out', nonlinearity='relu')

        elif init_name == 'kaiming normal' or init_name == 'he normal':
            weight_log = "Kaiming Normal (He Normal) weight initialization\n"
            for lin in model.lin:
                if lin == Linear:
                    init.kaiming_normal_(lin.weight, mode='fan_out', nonlinearity='relu')
                    init.kaiming_normal_(lin.bias, mode='fan_out', nonlinearity='relu')

        elif init_name == 'orthogonal':
            weight_log = "Orthogonal weight initialization : gain {0:4.2f}\n".format(gain)
            for lin in model.lin:
                if lin == Linear:
                    init.orthogonal_(lin.weight, gain=gain)
                    init.orthogonal_(lin.bias, gain=gain)
    except:
        import sys
        print(sys.exc_info())
        weight_log = "During weight initialization error occured. Default Initialization used\n"
        
    return  weight_log

# Convert generated scale_factor, pca to pytorch tensor format 
def _convert_to_tensor(inputs, logfile, scale_factor, pca):
    for item in inputs['atom_types']:
        if inputs['descriptor']['calc_scale']:
            max_plus_min  = torch.tensor(scale_factor[item][0,:])
            max_minus_min = torch.tensor(scale_factor[item][1,:])
            scale_factor[item] = [max_plus_min, max_minus_min] #To list format
            logfile.write('Convert {0} scale_factor to tensor\n'.format(item))
        if inputs['neural_network']['pca']:
            pca[item][0] = torch.tensor(pca[item][0])
            pca[item][1] = torch.tensor(pca[item][1])
            pca[item][2] = torch.tensor(pca[item][2])
            logfile.write('Convert {0} PCA to tensor\n'.format(item))

def _load_lammps_potential(inputs, logfile, model):
    logfile.write('Load parameters from lammps potential filename: {0}\n'.format(inputs["neural_network"]['continue']))
    potential_params = read_lammps_potential(inputs["neural_network"]['continue'])
    assert [item for item in model.keys].sort() == [item for item in potential_params.keys()].sort()
    for item in potential_params.keys():
        for name, lin in model.nets[item].lin.named_modules():
            if name in potential_params[item].keys():
                lin.weight = Parameter(torch.transpose(torch.tensor(potential_params[item][name]['weight']),-1,0))
                lin.bias = Parameter(torch.transpose(torch.tensor(potential_params[item][name]['bias']),-1,0))
 
