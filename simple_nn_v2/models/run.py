import torch
import numpy as np
from functools import partial
from sklearn.decomposition import PCA

from .neural_network import FCNDict, FCN
from .data_handler import TorchStyleDataset, FilelistDataset, my_collate
from .train import train, save_checkpoint




def run_model(inputs, logfile):
    #Set default type
    torch.set_default_dtype(torch.float64)

    # Initialize model
    model, optimizer, criterion, scale_factor, pca = _init_model(inputs, logfile)
    
    # Resume job if possible and load dataset & scale_factor, pca
    scale_factor, pca, train_dataset, valid_dataset, best_loss = _load_data(\
    inputs, logfile, model, optimizer, scale_factor, pca)

    # Convert scale_factor, pca to torch.tensor
    _convert_to_tensor(inputs, logfile, scale_factor, pca)

    # Load data loader
    train_loader, valid_loader = _load_collate(inputs, logfile, scale_factor, pca, train_dataset, valid_dataset)
    
    # Run training
    best_loss, best_epoch = _do_train(inputs, logfile, train_loader, valid_loader, model, optimizer, criterion, scale_factor, pca, best_loss) 

    # Best loss witten
    logfile.write('Best loss lammps potential written at {0} epoch\n'.format(best_epoch))


#Initialize model with input, and set default value scale, PCA
def _init_model(inputs, logfile):
    #Set default configuration
    logfile.write('Initialize pytorch model\n')
    model = {}
    for item in inputs['atom_types']:
        temp_nodes = [int(jtem) for jtem in inputs['neural_network']['nodes'].split('-')]
        #Extract number of symmetry functions in params
        with open(inputs['symmetry_function']['params'][item],'r') as f:
            tmp_symf = f.readlines()
            sym_num = len(tmp_symf)
        #Make NN for each element 
        model[item] = FCN(sym_num, temp_nodes, acti_func=\
        inputs['neural_network']['acti_func']) #Make nodes per elements
    model = FCNDict(model) #Macle full model with elementized dictionary model

    try: #Check avaiable CUDA (GPU)
        model.cuda()
        logfile.write("Use GPU(CUDA) machine \n")
    except:
        model.cpu()
        logfile.write("GPU is not available. Use CPU machine\n")
        pass

    if inputs['neural_network']['method'] == 'Adam':
        #Adam optimizer (Default)
        optimizer = torch.optim.Adam(model.parameters(), lr=inputs['neural_network']['learning_rate'],
     weight_decay=inputs['neural_network']['weight_decay'])
    else:
        #Other method should be implemented 
        pass

    criterion = torch.nn.MSELoss().cuda()

    scale_factor = None
    pca = None
    
    return  model, optimizer, criterion, scale_factor, pca


def _load_data(inputs, logfile, model, optimizer, scale_factor, pca):
    # Load checkpoint from resume (if necessary)
    if inputs["neural_network"]["resume"] is not None:
        logfile.write('Load pytorch model from {0}\n'.format(inputs["neural_network"]["resume"]))
        checkpoint = torch.load(inputs["neural_network"]["resume"])
        # Load model
        if not inputs["neural_network"]['clear_prev_network']:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg in optimizer.param_groups:
                pg['lr'] = inputs['neural_network']['learning_rate']
        if not inputs['neural_network']['clear_prev_status']:
            best_loss = checkpoint['best_loss']
            inputs['neural_network']['start_epoch'] = checkpoint['epoch']

        # Load scale file & PCA file
        if inputs['symmetry_function']['calc_scale']:
            scale_factor = checkpoint['scale_factor']
        if inputs['neural_network']['pca']:
            pca = checkpoint['pca']

    else: #Not resume, load scale_factor, pca pytorch savefile in preprocess
        #Set best loss as infinite
        best_loss = float('inf')
        if inputs['symmetry_function']['calc_scale']:
            scale_factor = torch.load('./scale_factor')
            logfile.write('Scale factor data loaded\n')

        if inputs['neural_network']['pca']:
            pca = torch.load('./pca')
            logfile.write('PCA data loaded\n')

    # Load dataset 
    train_dataset = FilelistDataset(inputs['symmetry_function']['train_list'])
    if not inputs['neural_network']['evaluate']:
        valid_dataset = FilelistDataset(inputs['symmetry_function']['valid_list']) 
    logfile.write('Train & Valid dataset loaded\n')

    #model, optimizer modified but not return
    return scale_factor, pca, train_dataset, valid_dataset, best_loss 



# Convert generated scale_factor, pca to pytorch tensor format 
def _convert_to_tensor(inputs, logfile, scale_factor, pca):
    for item in inputs['atom_types']:
        if inputs['symmetry_function']['calc_scale']:
            max_plus_min  = torch.tensor(scale_factor[item][0,:])
            max_minus_min = torch.tensor(scale_factor[item][1,:])
            scale_factor[item] = [max_plus_min, max_minus_min] #To list format
            logfile.write('Convert {0} scale_factor to tensor\n'.format(item))
        if inputs['neural_network']['pca']:
            pca[item][0] = torch.tensor(pca[item][0])
            pca[item][1] = torch.tensor(pca[item][1])
            pca[item][2] = torch.tensor(pca[item][2])
            logfile.write('Convert {0} PCA to tensor\n'.format(item))


def _load_collate(inputs, logfile, scale_factor, pca, train_dataset, valid_dataset):

    partial_collate = partial(
        my_collate, 
        atom_types=inputs['atom_types'], 
        scale_factor=scale_factor, 
        pca=pca, 
        pca_min_whiten_level=inputs['neural_network']['pca_min_whiten_level'],
        use_stress=inputs['neural_network']['use_stress'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=inputs['neural_network']['batch_size'], shuffle=True, collate_fn=partial_collate,
        num_workers=inputs['neural_network']['workers'], pin_memory=True)

    if not inputs['neural_network']['evaluate']:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=inputs['neural_network']['batch_size'], shuffle=False, collate_fn=partial_collate,
            num_workers=inputs['neural_network']['workers'], pin_memory=True)

    logfile.write('Train & Valid dataloader setted\n')
    return train_loader, valid_loader




def _do_train(inputs, logfile, train_loader, valid_loader, model, optimizer, criterion, scale_factor, pca, best_loss):

    #Check GPU (CUDA) available
    CUDA = torch.cuda.is_available()
    #Check use validation
    if float(inputs['symmetry_function']['valid_rate']) < 1E-6:
        valid = False
        logfile.write('Not use validation\n')
    else:
        valid = True
        logfile.write('Use validation\n')
    # Run training process
    if inputs['neural_network']['evaluate']:
        loss = train(inputs, logfile, train_loader, model, criterion=criterion, valid=valid, cuda=CUDA)
    else:
        for epoch in range(inputs['neural_network']['start_epoch'], inputs['neural_network']['total_iteration']):
            #Train model with train loader 
            train(inputs, logfile, train_loader, model, optimizer=optimizer, criterion=criterion, epoch=epoch, cuda=CUDA)
            #Calculate valid loss with valid loader
            loss = train(inputs, logfile, valid_loader, model, criterion=criterion, valid=valid)
            is_best = loss < best_loss
            best_loss = min(best_loss, loss)
           
           
            #Checkpoint in model traning !!
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'best_loss': best_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pca': pca,
                    'scale_factor': scale_factor,
                    #'scheduler': scheduler.state_dict()
                }, is_best)

            if is_best: #Save best lammps potential
                model.write_lammps_potential(filename ='./potential_saved_best_loss', inputs=inputs, scale_factor=scale_factor, pca=pca)
                best_epoch = epoch 
            if epoch % inputs['neural_network']['save_interval'] == 0:
                model.write_lammps_potential(filename ='./potential_saved_epoch_{0}'.format(epoch), inputs=inputs, scale_factor=scale_factor, pca=pca)
                logfile.write('Lammps potential written at {0} epoch\n'.format(epoch))
    return best_loss, best_epoch
