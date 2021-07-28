# Can't run right after train reference potential
# Because of demanding 'replica_list' file
# Also, 'potential_saved' file

import torch
from simple_nn_v2.models import data_handler
from simple_nn_v2.models import optimizers
from simple_nn_v2.models.run import _get_torch_device, _set_pararrelism, _load_model_weights_and_optimizer_from_checkpoint, _load_scale_factor_and_pca, train_model
from simple_nn_v2.models import utils
from simple_nn_v2.models import neural_network
from simple_nn_v2.models import run

def replica_run(inputs, logfile):
    if inputs['neural_network']['double_precision']:
        torch.set_default_dtype(torch.float64)
    device = _get_torch_device(inputs)
    _set_pararrelism(inputs, logfile)

    if inputs['replica']['add_NNP_ref']:
        # Todo: Log for start save_atomic_e
        model = neural_network._initialize_model_and_weights(inputs, logfile, device)
        optimizer = optimizers._initialize_optimizer(inputs, model)
        checkpoint = run._load_model_weights_and_optimizer_from_checkpoint(inputs, logfile, model, optimizer, device)
        scale_factor, pca = _load_scale_factor_and_pca(inputs, logfile, checkpoint)

        data_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='add_NNP_ref')
        save_atomic_E(inputs, logfile, model, data_loader, device)
    
    if inputs['replica']['train_atomic_E']:
        # Todo: Log for train save_atomic_e
        model = neural_network._initialize_model_and_weights(inputs, logfile, device)
        optimizer = optimizers._initialize_optimizer(inputs, model)
        scale_factor, pca = _load_scale_factor_and_pca(inputs, logfile, checkpoint=None)
        criterion = torch.nn.MSELoss(reduction='none').to(device=device)

        train_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='atomic_E_train')
        valid_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='atomic_E_valid')
        labeled_train_loader = data_handler._load_labeled_dataset(inputs, logfile, scale_factor, pca, device, mode='atomic_E_train')
        labeled_valid_loader = data_handler._load_labeled_dataset(inputs, logfile, scale_factor, pca, device, mode='atomic_E_valid')

        train_model(inputs, logfile, model, optimizer, criterion, scale_factor, pca, device, float('inf'),\
            train_loader, valid_loader, labeled_train_loader, labeled_valid_loader, atomic_e=True)

def save_atomic_E(inputs, logfile, model, data_loader, device):
    #Save NNP energy, force, DFT energy, force to use it
    model.eval()
    non_block = False if (device == torch.device('cpu')) else True
 
    for i, item in enumerate(data_loader):
        print(item.keys())
        n_batch = item['E'].size(0) 
        n_type = dict()
        for atype in inputs['atom_types']:
            n_type[atype] = 0
        x, atomic_E, _, _ = utils.calculate_E(inputs['atom_types'], item, model, device, non_block)
                
        #Save    
        for f in range(n_batch): 
            pt_dict = torch.load(item['filename'][f])
            if 'filename' in pt_dict.keys():
                del pt_dict['filename'] 
            if 'atomic_E' in pt_dict.keys():
                del pt_dict['atomic_E'] 
            pt_dict['atomic_E'] = dict()
            for atype in inputs['atom_types']:
                if (x[atype].size(0) != 0) and (item['n'][atype][f].item() != 0):
                    file_atomic_E = atomic_E[atype][f][n_type[atype]:n_type[atype] + item['n'][atype][f]]
                    pt_dict['atomic_E'][atype] = file_atomic_E.cpu().detach()
                    n_type[atype] += item['n'][atype][f] 
            torch.save(pt_dict, item['filename'][f])
