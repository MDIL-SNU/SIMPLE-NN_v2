import torch
from torch.optim.lr_scheduler import ExponentialLR
import time, os
from tqdm import tqdm
from simple_nn_v2.models import neural_network, loss, data_handler, optimizers, logger

#Main function that train neural network
def train(inputs, logfile):
    start_time = time.time()

    if inputs['neural_network']['double_precision']:
        torch.set_default_dtype(torch.float64)
    device = _get_torch_device(inputs)
    _set_pararrelism(inputs, logfile)

    model = neural_network._initialize_model_and_weights(inputs, logfile, device)
    optimizer = optimizers._initialize_optimizer(inputs, model)
    criterion = torch.nn.MSELoss(reduction='none').to(device=device)

    checkpoint = _load_model_weights_and_optimizer_from_checkpoint(inputs, logfile, model, optimizer, device)
    scale_factor, pca = _load_scale_factor_and_pca(inputs, logfile, checkpoint)
    loss = _set_initial_loss(inputs, logfile, checkpoint)

    if inputs['neural_network']['train']:
        train_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='train', gdf=inputs['neural_network']['atomic_weights'])
        if os.path.exists(inputs['neural_network']['valid_list']):
            valid_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='valid', gdf=False)
        else:
            valid_loader = None
        train_model(inputs, logfile, model, optimizer, criterion, scale_factor, pca, device, loss, train_loader, valid_loader)

    if inputs['neural_network']['test']:
        test_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='test', gdf=False)
        test_model(inputs, logfile, model, optimizer, criterion, device, test_loader)

    if inputs['neural_network']['add_NNP_ref']:
        ref_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='add_NNP_ref')
        save_atomic_E(inputs, logfile, model, ref_loader, device)
        logfile.write("Adding NNP Energy to pt files Done\n")

    if inputs['neural_network']['train_atomic_E']:
        train_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='atomic_E_train')
        if os.path.exists(inputs['neural_network']['valid_list']):
            valid_loader = data_handler._load_dataset(inputs, logfile, scale_factor, pca, device, mode='atomic_E_valid')
        train_model(inputs, logfile, model, optimizer, criterion, scale_factor, pca, device, float('inf'),\
            train_loader, valid_loader, atomic_e=True)
    logfile.write(f"Elapsed time in training. {time.time()-start_time:10} s.\n")

def _get_torch_device(inputs):
    if inputs['neural_network']['load_data_to_gpu'] and torch.cuda.is_available():
        cuda_num = inputs['neural_network']['GPU_number']
        device = 'cuda'+':'+str(cuda_num) if cuda_num else 'cuda'
    else:
        device = 'cpu'
    return torch.device(device)

def _set_pararrelism(inputs, logfile):
    if inputs['neural_network']['intra_op_parallelism_threads'] != 0:
        torch.set_num_threads(inputs['neural_network']['intra_op_parallelism_threads'])
    if inputs['neural_network']['inter_op_parallelism_threads'] != 0:
        torch.set_num_interop_threads(inputs['neural_network']['inter_op_parallelism_threads'])
    logfile.write("Parallelism intra_thread : {0} inter_thread : {1}\n".format(torch.get_num_threads(), torch.get_num_interop_threads()))

def _load_model_weights_and_optimizer_from_checkpoint(inputs, logfile, model, optimizer, device):
    checkpoint = None

    if inputs['neural_network']['continue'] == 'weights': # load lammps potential
        potential_params = neural_network.read_lammps_potential('./potential_saved')
        for element in potential_params.keys():
            for name, lin in model.nets[element].lin.named_modules():
                if name in potential_params[element].keys():
                    lin.weight.data = torch.transpose(torch.tensor(potential_params[element][name]['weight']).to(device=device), -1, 0)
                    lin.bias.data = torch.transpose(torch.tensor(potential_params[element][name]['bias']).to(device=device), -1, 0)
    elif inputs['neural_network']['continue']: # load pytorch type checkpoint
        checkpoint = torch.load(inputs['neural_network']['continue'], map_location=device)
        model.load_state_dict(checkpoint['model'])
        logfile.write("Load pytorch model from [{0}]\n".format(inputs['neural_network']['continue']))

        if not inputs['neural_network']['clear_prev_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg in optimizer.param_groups:
                pg['lr'] = inputs['neural_network']['learning_rate']

        if not inputs['neural_network']['clear_prev_status']:
            inputs['neural_network']['start_epoch'] = checkpoint['epoch']

    return checkpoint

def _load_scale_factor_and_pca(inputs, logfile, checkpoint):
    scale_factor = None
    pca = None

    if checkpoint:  # load from checkpoint file
        if inputs['neural_network']['scale']:
            scale_factor = checkpoint['scale_factor']
            logfile.write("Load scale factor from checkpoint\n")
        if inputs['neural_network']['pca']:
            pca = checkpoint['pca']
            logfile.write("Load pca from checkpoint\n")
    else:  # load from 'scale_factor', 'pca' file
        if inputs['neural_network']['scale']:
            if type(inputs['neural_network']['scale']) is not bool:
                scale_factor = torch.load(inputs['scale']['scale'])
            else:
                scale_factor = torch.load('./scale_factor')
        if inputs['neural_network']['pca']:
            if type(inputs['neural_network']['pca']) is not bool:
                pca = torch.load(inputs['neural_network']['pca'])
            else:
                pca = torch.load('./pca')
        _convert_to_tensor(inputs, logfile, scale_factor, pca)

    return scale_factor, pca

def _convert_to_tensor(inputs, logfile, scale_factor, pca):
    device = _get_torch_device(inputs)
    for element in inputs['atom_types']:
        if scale_factor:
            max_plus_min  = torch.tensor(scale_factor[element][0], device=device)
            max_minus_min = torch.tensor(scale_factor[element][1], device=device)
            scale_factor[element] = [max_plus_min, max_minus_min]
        if pca:
            pca[element][0] = torch.tensor(pca[element][0], device=device)
            pca[element][1] = torch.tensor(pca[element][1], device=device)
            pca[element][2] = torch.tensor(pca[element][2], device=device)
            #cutoff_for_log = inputs['weight_modifier']['params'][item]['c']

def _set_initial_loss(inputs, logfile, checkpoint):
    loss = float('inf')
    if checkpoint and not inputs['neural_network']['clear_prev_status']:
        loss = checkpoint['loss']
        logfile.write("Load previous loss : {0:6.2e}\n".format(checkpoint['loss']))

    return loss

def train_model(inputs, logfile, model, optimizer, criterion, scale_factor, pca, device, best_loss, train_loader, valid_loader, atomic_e=False):
    struct_labels = _get_structure_labels(train_loader, valid_loader)
    dtype = torch.get_default_dtype()
    non_block = False if (device == torch.device('cpu')) else True

    max_len = len(train_loader)
    total_epoch = int(inputs['neural_network']['total_epoch'])
    total_iter = int(inputs['neural_network']['total_epoch'] * max_len)
    batch_size = 'full_batch' if inputs['neural_network']['full_batch'] else inputs['neural_network']['batch_size']
    logfile.write("Total training iteration : {0} , epoch : {1}, batch number : {2}, batch size : {3}, workers : {4}\n"\
    .format(total_iter, total_epoch, max_len, batch_size, inputs['neural_network']['workers']))

    criteria_dict = _set_stop_rmse_criteria(inputs, logfile)
    best_epoch = inputs['neural_network']['start_epoch']

    if inputs['neural_network']['lr_decay']:
        scheduler = ExponentialLR(optimizer=optimizer, gamma=inputs['neural_network']['lr_decay'])

    start_time = time.time()
    for epoch in tqdm(range(inputs['neural_network']['start_epoch'], total_epoch+1)):
        # Main loop for one epoch
        train_epoch_result = progress_epoch(inputs, train_loader, struct_labels, model, optimizer, criterion, epoch, dtype, device, non_block, valid=False, atomic_e=atomic_e)
        train_loss = train_epoch_result['losses'].avg
        if valid_loader:
            valid_epoch_result = progress_epoch(inputs, valid_loader, struct_labels, model, optimizer, criterion, epoch, dtype, device, non_block, valid=True, atomic_e=atomic_e)
            loss = valid_epoch_result['losses'].avg
        else:
            valid_epoch_result = None
            loss = train_loss

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='checkpoint_bestmodel.pth.tar')
            model.write_lammps_potential(filename='./potential_saved_bestmodel', inputs=inputs, scale_factor=scale_factor, pca=pca)

        if (epoch % inputs['neural_network']['show_interval'] == 0):
            if inputs['neural_network']['accurate_train_rmse']:
                recalc_epoch_result = progress_epoch(inputs, train_loader, struct_labels, model, optimizer, criterion, epoch, dtype, device, non_block, valid=True, atomic_e=atomic_e)
                update_recalc_results(train_epoch_result, recalc_epoch_result)

            total_time = time.time() - start_time
            logger._show_avg_rmse(inputs, logfile, epoch, optimizer.param_groups[0]['lr'], total_time, train_epoch_result, valid_epoch_result)
            if inputs['neural_network']['print_structure_rmse']:
                logger._show_structure_rmse(inputs, logfile, train_epoch_result, valid_epoch_result)

        # save checkpoint for each checkpoint_interval
        if inputs['neural_network']['checkpoint_interval'] and (epoch % inputs['neural_network']['checkpoint_interval'] == 0):
            save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename=f'checkpoint_epoch_{epoch}.pth.tar')
            model.write_lammps_potential(filename='./potential_saved_epoch_{0}'.format(epoch), inputs=inputs, scale_factor=scale_factor, pca=pca)
            logfile.write("Lammps potential written at {0} epoch\n".format(epoch))
        elif not inputs['neural_network']['checkpoint_interval']:
            save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='checkpoint_latest.pth.tar')
            model.write_lammps_potential(filename='./potential_saved_latest', inputs=inputs, scale_factor=scale_factor, pca=pca)

        #Break if energy, force, stress is under their criteria
        if criteria_dict:
            breaksignal = True
            for err_type in criteria_dict.keys():
                criteria_dict[err_type][0] = valid_epoch_result['tot_'+err_type].sqrt_avg if valid_loader else train_epoch_result['tot_'+err_type].sqrt_avg
                if criteria_dict[err_type][0] > criteria_dict[err_type][1]:
                    breaksignal = False
            if breaksignal:
                logfile.write("Break point reached. Terminating traning model\n")
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='checkpoint_criterion.pth.tar')
                model.write_lammps_potential(filename='./potential_saved_criterion', inputs=inputs, scale_factor=scale_factor, pca=pca)
                logfile.write("checkpoint_criterion.pth.tar & potential_saved_criterion written\n")
                break

        if inputs['neural_network']['lr_decay']:
            scheduler.step()
        # soft termination 
        if os.path.exists('./stoptrain'):
                os.remove('./stoptrain')
                logfile.write("Stop traning by ./stoptrain file.\n")
                save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='checkpoint_stoptrain.pth.tar')
                model.write_lammps_potential(filename='./potential_saved_stoptrain', inputs=inputs, scale_factor=scale_factor, pca=pca)
                logfile.write("checkpoint_stoptrain.pth.tar & potential_saved_stoptrain written\n")
                break

        logfile.flush()

    logfile.write("Best loss lammps potential written at {0} epoch\n".format(best_epoch))

#Evalutaion model & save result  -> result_saved
def test_model(inputs, logfile, model, optimizer, criterion, device, test_loader):
    struct_labels = _get_structure_labels(test_loader, valid_loader=None)
    logfile.write("Evaluation(Testing) model \n")
    dtype = torch.get_default_dtype()
    non_block = False if (device == torch.device('cpu')) else True

    use_force = inputs['neural_network']['use_force']
    use_stress = inputs['neural_network']['use_stress']
    epoch_result = logger._init_meters(struct_labels, use_force, use_stress, atomic_e=False)
    res_dict = _initialize_test_result_dict(inputs)
    model.eval()

    for i, item in enumerate(test_loader):
        n_batch = item['E'].size(0)
        _, calc_results = loss.calculate_batch_loss(inputs, item, model, criterion, device, non_block, epoch_result, False, dtype, use_force, use_stress, atomic_e=False)
        #TODO: Add E, F, S to pt file as option  "NNP_to_pt"
        _update_calc_results_in_results_dict(n_batch, res_dict, item, calc_results, use_force, use_stress)

    torch.save(res_dict, 'test_result')
    logfile.write(f"DFT, NNP result saved at 'result_saved'\n")

    logger._show_avg_rmse(inputs, logfile, 0, 0, 0, epoch_result, None)

    if inputs['neural_network']['print_structure_rmse']: 
        logger._show_structure_rmse(inputs, logfile, epoch_result, None)

#Main loop for calculations 
def progress_epoch(inputs, data_loader, struct_labels, model, optimizer, criterion, epoch, dtype, device, non_block, valid=False, atomic_e=False):
    use_force = inputs['neural_network']['use_force'] if not atomic_e else False
    use_stress = inputs['neural_network']['use_stress'] if not atomic_e else False
    epoch_result = logger._init_meters(struct_labels, use_force, use_stress, atomic_e)

    weighted = False if valid else True
    back_prop = False if valid else True
    model.eval() if valid else model.train()

    end = time.time()
    for i, item in enumerate(data_loader):
        epoch_result['data_time'].update(time.time() - end) # save data loading time
        batch_loss, _ = loss.calculate_batch_loss(inputs, item, model, criterion, device, non_block, epoch_result, weighted, dtype, use_force, use_stress, atomic_e)
        if back_prop: 
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        epoch_result['batch_time'].update(time.time() - end) # save batch calculation time
        end = time.time()

    return epoch_result

def update_recalc_results(train_result, recalc_result):
    train_result['losses'] = recalc_result['losses']
    train_result['e_err'] = recalc_result['e_err']
    train_result['tot_e_err'] = recalc_result['tot_e_err']
    if 'f_err' in train_result.keys():
        train_result['f_err'] = recalc_result['f_err']
        train_result['tot_f_err'] = recalc_result['tot_f_err']
    if 's_err' in train_result.keys():
        train_result['s_err'] = recalc_result['s_err']
        train_result['tot_s_err'] = recalc_result['tot_s_err']

    train_result['batch_time'].val += recalc_result['batch_time'].val
    train_result['data_time'].val += recalc_result['data_time'].val

def save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename):
    state ={'epoch': epoch + 1,
            'loss': loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pca': pca,
            'scale_factor': scale_factor,
            #'scheduler': scheduler.state_dict()
            }
    torch.save(state, filename)

def _get_structure_labels(train_loader, valid_loader=None):
    labels = []
    for i, item in enumerate(train_loader):
        for str_type in item['struct_type']:
            if str_type not in labels:
                labels.append(str_type)
    if valid_loader:
        for i, item in enumerate(valid_loader):
            for str_type in item['struct_type']:
                if str_type not in labels:
                    labels.append(str_type)
    return sorted(labels)

def _set_stop_rmse_criteria(inputs, logfile):
    criteria_dict = dict()
    if inputs['neural_network']['energy_criteria']:
        criteria_dict['e_err'] = [float('inf') , float(inputs['neural_network']['energy_criteria'])]
        logfile.write("Energy criteria used : {0:4}  \n" .format(float(inputs['neural_network']['energy_criteria'])))
    if inputs['neural_network']['force_criteria'] and inputs['neural_network']['use_force']:
        criteria_dict['f_err'] = [float('inf'), float(inputs['neural_network']['force_criteria'])]
        logfile.write("Force criteria used : {0:4}  \n" .format(float(inputs['neural_network']['force_criteria'])))
    if inputs['neural_network']['stress_criteria'] and inputs['neural_network']['use_stress']:
        criteria_dict['s_err'] = [float('inf'), float(inputs['neural_network']['stress_criteria'])]
        logfile.write("Stress criteria used : {0:4}  \n" .format(float(inputs['neural_network']['stress_criteria'])))

    if len(criteria_dict.keys()) == 0:
        criteria_dict = None

    return criteria_dict

def _initialize_test_result_dict(inputs):
    res_dict = {'DFT_E': list(), 'NN_E': list(), 'N': list()}
    if inputs['neural_network']['use_force']:
        res_dict['DFT_F'] = list()
        res_dict['NN_F'] = list()
    if inputs['neural_network']['use_stress']:
        res_dict['DFT_S'] = list()
        res_dict['NN_S'] = list()

    return res_dict

def _update_calc_results_in_results_dict(n_batch, res_dict, item, calc_results, use_force, use_stress):
    for n in range(n_batch):
        res_dict['N'].append(item['tot_num'][n].item())
        res_dict['NN_E'].append(calc_results['E'][n].item())
        res_dict['DFT_E'].append(item['E'][n].item())

    if use_force:
        batch_idx = 0
        for n in range(n_batch):
            tmp_idx = item['tot_num'][n].item()
            res_dict['NN_F'].append(calc_results['F'][batch_idx:(batch_idx+tmp_idx)].cpu().detach().numpy())
            res_dict['DFT_F'].append(item['F'][batch_idx:(batch_idx+tmp_idx)].cpu().detach().numpy())
            batch_idx += tmp_idx

    if use_stress:
        batch_idx = 0
        for n in range(n_batch):
            tmp_idx = 6
            res_dict['NN_S'].append(calc_results['S'][batch_idx:(batch_idx+tmp_idx)].cpu().detach().numpy())
            res_dict['DFT_S'].append(item['S'][batch_idx:(batch_idx+tmp_idx)].cpu().detach().numpy())
            batch_idx += tmp_idx

def save_atomic_E(inputs, logfile, model, data_loader, device):
    model.eval()
    non_block = False if (device == torch.device('cpu')) else True

    for i, item in enumerate(data_loader):
        print(item.keys())
        n_batch = item['E'].size(0)
        n_type = dict()
        for atype in inputs['atom_types']:
            n_type[atype] = 0
        x, atomic_E, _, _ = loss.calculate_E(inputs['atom_types'], item, model, device, non_block)

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
