import torch
import shutil
import time
from ase import units
from simple_nn_v2.utils.Logger import AverageMeter, ProgressMeter, TimeMeter, StructureMeter
from simple_nn_v2.models.data_handler import FilelistDataset, _make_dataloader

#This function train NN 
def train(inputs, logfile, data_loader, model, optimizer=None, criterion=None, scheduler=None, epoch=0, valid=False, err_dict=None,start_time=None, test=False):
    
    save_result = inputs['neural_network']['save_result']
    show_interval = inputs['neural_network']['show_interval']

    progress, progress_dict, = _init_meters(model, data_loader, optimizer, epoch, 
    valid, inputs['neural_network']['use_force'], inputs['neural_network']['use_stress'], test)

    end = time.time()
    max_len = len(data_loader)
    
    #Training part
    for i, item in enumerate(data_loader):
        progress_dict['data_time'].update(time.time() - end)
        
        loss = _loop_for_loss(inputs, item, model, criterion, progress_dict, struct_weight=(not valid))

        #Training part
        if not valid and not test: 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        progress_dict['batch_time'].update(time.time() - end)
        end = time.time()
    if test:
        logfile.write(progress.test())
    elif epoch % show_interval == 0:
        progress_dict['total_time'].update(time.time() - start_time)
        progress.display_epoch()
        logfile.write(progress.log_epoch())
    
    # After one epoch load err_dict & use_force, use_stress included err_dict
    if err_dict:
        for err_type in err_dict.keys():
            err_dict[err_type][0] = progress_dict[err_type].avg

    return progress_dict['losses'].avg
    
def _init_meters(model, data_loader, optimizer, epoch, valid, use_force, use_stress, test):
    ## Setting LOG with progress meter
    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('loss', ':8.4e')
    e_err = AverageMeter('E err', ':6.4e', sqrt=True)
    total_time = TimeMeter('total time', ':8.4e')
    progress_list = [losses, e_err]
    progress_dict = {'batch_time': batch_time, 'data_time': data_time, 'losses': losses, 'e_err': e_err, 'total_time': total_time} 
    
    if use_force:
        f_err = AverageMeter('F err', ':6.4e', sqrt=True)
        progress_list.append(f_err)
        progress_dict['f_err'] = f_err
    if use_stress:
        s_err = AverageMeter('S err', ':6.4e', sqrt=True)
        progress_list.append(s_err)
        progress_dict['s_err'] = s_err
    if not test:  #Use training, validation loop
        progress_list.append(batch_time)
        progress_list.append(data_time)
        progress_list.append(total_time)

    #Prefix
    if test:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Test    : ",
        )
        model.eval()
    elif valid:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix=f"Valid :[{epoch:6}]",
        )
        model.eval()
    else:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix=f"Epoch :[{epoch:6}]",
            suffix=f"lr: {optimizer.param_groups[0]['lr']:6.4e}"
        )
        model.train()

    return progress, progress_dict

#Show structure rmse
def _show_structure_rmse(inputs, logfile, train_struct_dict, valid_struct_dict, model, optimizer=None, criterion=None, cuda=False, test=False):
    for t_key in train_struct_dict.keys():
        log_train = _struct_log(inputs, train_struct_dict[t_key], model, optimizer=optimizer, criterion=criterion, cuda=cuda, test=test)
        log_train = "[{0:8}] ".format(t_key) + log_train

        if valid_struct_dict[t_key]:
            log_valid = _struct_log(inputs, valid_struct_dict[t_key], model, valid=True, optimizer=optimizer, criterion=criterion, cuda=cuda)
            log_valid = "\n[{0:8}] ".format(t_key) + log_valid
        else:
            log_valid = ""
        outdict = log_train + log_valid

        print(outdict)
        logfile.write(outdict+'\n')

#Generate structure log string
def _struct_log(inputs, data_loader, model, valid=False, optimizer=None, criterion=None, cuda=False, test=False):

    dtype = torch.get_default_dtype()

    losses = StructureMeter('loss', ':8.4e')
    e_err = StructureMeter('E err', ':6.4e', sqrt=True)

    progress_list = [losses, e_err]
    progress_dict = {'losses': losses, 'e_err': e_err} 

    if inputs['neural_network']['use_force']:
        f_err = StructureMeter('F err', ':6.4e', sqrt=True)
        progress_list.append(f_err)
        progress_dict['f_err'] = f_err
    if inputs['neural_network']['use_stress']:
        s_err = StructureMeter('S err', ':6.4e', sqrt=True)
        progress_list.append(s_err)
        progress_dict['s_err'] = s_err
    if test:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Test : ",
        )

    elif valid:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Valid : ",
        )
    else:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Train : ",
        )

    model.eval()
    #Evaluation part
    for i,item in enumerate(data_loader):
        loss = _loop_for_loss(inputs, item, model, criterion, progress_dict)
    
    return progress.string()

#Traning loop for loss 
def _loop_for_loss(inputs, item, model, criterion, progress_dict, struct_weight=False):

    dtype = torch.get_default_dtype()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = 0.
    e_loss = 0.
    n_batch = item['E'].size(0) 
   
    # Since the shape of input and intermediate state during forward is not fixed,
    # forward process is done by structure by structure manner.
    x = dict()

    if struct_weight:
        weight = item['struct_weight']
    else:
        weight = torch.ones(n_batch,device=device)
    if inputs['neural_network']['use_force']:
        F = item['F'].type(dtype)
    if inputs['neural_network']['use_stress']:
        S = item['S'].type(dtype)


    E_ = 0.
    F_ = 0.
    S_ = 0.
    n_atoms = 0.
 
    #Loop for calculating energy
    for atype in inputs['atom_types']:
        x[atype] = item['x'][atype].requires_grad_(True)
        if x[atype].size(0) != 0:
            E_ += torch.sum(torch.sparse.DoubleTensor(
                item['sp_idx'][atype].long(), 
                model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                item['sp_idx'][atype].size(1))).to_dense(), axis=1)
        n_atoms += item['n'][atype]
    #Energy loss + structure_weight
    e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype) / n_atoms)
    print_e_loss = torch.mean(e_loss * weight)
    e_loss = torch.mean(e_loss * weight)

    #Loop for force, stress
    if inputs['neural_network']['use_force'] or inputs['neural_network']['use_stress']:
        #Loop for elements type
        for atype in inputs['atom_types']:
            if x[atype].size(0) != 0:
                dEdG = torch.autograd.grad(torch.sum(E_), x[atype], create_graph=True)[0]

                tmp_force = list()
                tmp_stress = list()
                tmp_idx = 0
                
                for n,ntem in enumerate(item['n'][atype]):
                    if inputs['neural_network']['use_force']: #force loop + structure_weight
                        if ntem != 0:
                            tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]))
                        else:
                            tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1],device=device))

                    if inputs['neural_network']['use_stress']: #stress loop
                        if ntem != 0:
                            tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                        else:
                            tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1],device=device))
                    #Index sum
                    tmp_idx += ntem

                if inputs['neural_network']['use_force']:
                    F_ -= torch.cat(tmp_force, axis=0)

                if inputs['neural_network']['use_stress']:
                    S_ -= torch.cat(tmp_stress, axis=0) / units.GPa * 10

        #Force loss part 
        if inputs['neural_network']['use_force']:
            if inputs['neural_network']['force_diffscale']:
                # check the scale value: current = norm(force difference)
                # Force different scaling : larger force difference get higher weight !!
                force_diffscale = torch.sqrt(torch.norm(F_ - F, dim=1, keepdim=True).detach())
                f_loss = criterion(force_diffscale * F_, force_diffscale * F)
                print_f_loss = torch.mean(criterion(F_, F))
                #batch_idx = 0
                #for n in range(n_batch): #Make structure_weighted force
                #    tmp_idx = item['tot_num'][n].item()
                #    f_loss[batch_idx:(batch_idx+tmp_idx)] = f_loss[batch_idx:(batch_idx+tmp_idx)]*weight[n].item()
                #    batch_idx += tmp_idx 
                f_loss = torch.mean(f_loss)
            else:
                f_loss = criterion(F_, F)
                print_f_loss = torch.mean(f_loss)
                batch_idx = 0
                for n in range(n_batch): #Make structure_weighted force
                    tmp_idx = item['tot_num'][n].item()
                    f_loss[batch_idx:(batch_idx+tmp_idx)] = f_loss[batch_idx:(batch_idx+tmp_idx)] * weight[n].item()
                    batch_idx += tmp_idx 
                f_loss = torch.mean(f_loss)
            loss += inputs['neural_network']['force_coeff'] * f_loss
            progress_dict['f_err'].update(print_f_loss.detach().item(), F_.size(0))

        #Stress loss part
        if inputs['neural_network']['use_stress']:
            s_loss = criterion(S_, S)
            print_s_loss = torch.mean(s_loss)
            batch_idx = 0
            for n in range(n_batch): #Make structure_weighted force
                tmp_idx = item['tot_num'][n].item()
                s_loss[batch_idx:(batch_idx+tmp_idx)] = s_loss[batch_idx:(batch_idx+tmp_idx)] * weight[n].item()
                batch_idx += tmp_idx 
            s_loss = torch.mean(s_loss)
            loss += inputs['neural_network']['stress_coeff'] * s_loss
            progress_dict['s_err'].update(print_s_loss.detach().item(), n_batch)

    #Energy loss part
    loss = loss + inputs['neural_network']['energy_coeff'] * e_loss
    progress_dict['e_err'].update(print_e_loss.detach().item(), n_batch)
    loss = inputs['neural_network']['loss_scale'] * loss
    progress_dict['losses'].update(loss.detach().item(), n_batch)
    return loss

def save_checkpoint(epoch, loss, model, optimizer, pca, scale_factor, filename='checkpoint.pth.tar'):
    state ={'epoch': epoch + 1,
            'loss': loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pca': pca,
            'scale_factor': scale_factor,
            #'scheduler': scheduler.state_dict()
            }
    torch.save(state, filename)


def _save_nnp_result(inputs, model, train_loader, valid_loader):
    #Save NNP energy, force, DFT energy, force to use it
    model.eval()
    #Set output
    res_dict = {'DFT': dict(), 'NNP': dict()}
    for datatype in ['train', 'valid']:
        res_dict['DFT'][datatype] = {'E': list(), 'F': list()}
        res_dict['NNP'][datatype] = {'E': list(), 'F': list()}
        res_dict['tot_num'] = list()
   
    _loop_to_save(inputs, model, train_loader, res_dict, save_dict='train')
    if valid_loader:
        _loop_to_save(inputs, model, valid_loader, res_dict, save_dict='valid')

    return res_dict

def _loop_to_save(inputs, model, dataset_loader, res_dict, save_dict='train'):
    #Set device
    cuda = torch.cuda.is_available()
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    dtype = torch.get_default_dtype()
 
    for i, item in enumerate(dataset_loader):
        x = dict()
        E_ = 0.
        F_ = 0.
        n_atoms = 0.
        n_batch = item['E'].size(0) 
        #Loop
        for atype in inputs['atom_types']:
            x[atype] = item['x'][atype].requires_grad_(True)
            #x[atype] = item['x'][atype].to(device=device, non_blocking=cuda).requires_grad_(True)
            if x[atype].size(0) != 0:
                atomic_E = torch.sparse.DoubleTensor(
                    item['sp_idx'][atype].long(), 
                    #item['sp_idx'][atype].long().to(device=device, non_blocking=cuda), 
                    model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                    item['sp_idx'][atype].size(1))).to_dense()
                E_ += torch.sum(atomic_E, axis = 1)
            n_atoms += item['n'][atype]
            #n_atoms += item['n'][atype].to(device=device, non_blocking=cuda)
        
        for n in range(n_batch):
            res_dict['tot_num'].append(item['tot_num'][n].item())
            res_dict['NNP'][save_dict]['E'].append(E_[n].item())
            res_dict['DFT'][save_dict]['E'].append(item['E'][n].item())
       
        if inputs['neural_network']['use_force']:
            F = item['F'].type(dtype)
            #F = item['F'].type(dtype).to(device=device, non_blocking=cuda)
            for atype in inputs['atom_types']:
                if x[atype].size(0) != 0:
                    dEdG = torch.autograd.grad(torch.sum(E_), x[atype], create_graph=True)[0]
                    tmp_force = list()
                    tmp_idx = 0
                    for n, ntem in enumerate(item['n'][atype]):
                        if ntem != 0:
                            tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]))
                            #tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n].to(device=device, non_blocking=cuda), dEdG[tmp_idx:(tmp_idx + ntem)]))
                        else:
                            tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1],device=device))
                            #tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]).to(device=device, non_blocking=cuda))
                        tmp_idx += ntem
                    F_ -= torch.cat(tmp_force, axis=0)
            batch_idx = 0

            for n in range(n_batch):
                tmp_idx = item['tot_num'][n].item()
                res_dict['NNP'][save_dict]['F'].append(F_[batch_idx:(batch_idx+tmp_idx)].cpu().detach().numpy())
                res_dict['DFT'][save_dict]['F'].append(item['F'][batch_idx:(batch_idx+tmp_idx)].cpu().detach().numpy())
                batch_idx += tmp_idx 


def _save_atomic_E(inputs, logfile, model, train_loader, valid_loader):
    #Save NNP energy, force, DFT energy, force to use it
    model.eval()
    _loop_to_save_atomic_E(inputs, model, train_loader)
    if valid_loader:
        _loop_to_save_atomic_E(inputs, model, valid_loader)

def _loop_to_save_atomic_E(inputs, model, dataset_loader):
    #Set device
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    dtype = torch.get_default_dtype()
 
    for i, item in enumerate(dataset_loader):
        x = dict()
        dx = dict()
        da = dict()
        n_type = dict()
        atomic_E = dict()
        n_atoms = 0.
        n_batch = item['E'].size(0) 
        #Loop
        for atype in inputs['atom_types']:
            n_type[atype] = 0
            x[atype] = item['x'][atype].requires_grad_(True)
            #x[atype] = item['x'][atype].to(device=device, non_blocking=cuda).requires_grad_(False)
            if x[atype].size(0) != 0:
                atomic_E[atype] = torch.sparse.DoubleTensor(
                    item['sp_idx'][atype].long(), 
                    #item['sp_idx'][atype].long().to(device=device, non_blocking=cuda), 
                    model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                    item['sp_idx'][atype].size(1))).to_dense()
        #Save    
        for f in range(n_batch): 
            pt_dict = torch.load(item['filename'][f])
            del pt_dict['filename'] 
            pt_dict['atomic_E'] = dict()
            for atype in inputs['atom_types']:
                if (x[atype].size(0) != 0) and (item['n'][atype][f].item() != 0):
                    file_atomic_E = atomic_E[atype][f][n_type[atype]:n_type[atype] + item['n'][atype][f]]
                    pt_dict['atomic_E'][atype] = file_atomic_E
                    n_type[atype] += item['n'][atype][f] 
            torch.save(pt_dict, item['filename'][f])

#Not use
def _save_dft_info(inputs, train_dataset, valid_dataset):
    dft_dict = { 
        'total': {
            'energy': list(),
            'force': list(),
        },
        'train': {
            'energy': list(),
            'force': list(),
        },
        'valid': {
            'energy': list(),
            'force': list(),
        }
    }
    for _, item in enumerate(train_dataset):
       dft_dict['total']['energy'].append(item['E'].item())
       dft_dict['train']['energy'].append(item['E'].item())
       if inputs['neural_network']['use_force']:
           dft_dict['total']['force'].append(item['F'])
           dft_dict['train']['force'].append(item['F'])
    if valid_dataset:
        for _, item in enumerate(valid_dataset):
           dft_dict['total']['energy'].append(item['E'].item())
           dft_dict['valid']['energy'].append(item['E'].item())
           if inputs['neural_network']['use_force']:
               dft_dict['total']['force'].append(item['F'])
               dft_dict['valid']['force'].append(item['F'])
    return dft_dict

def _loop_for_loss_backup(inputs, item, model, criterion, progress_dict, struct_weight=False):

    dtype = torch.get_default_dtype()

    loss = 0.
    e_loss = 0.
    n_batch = item['E'].size(0) 
    #Set device
    cuda = torch.cuda.is_available()
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
   
    # Since the shape of input and intermediate state during forward is not fixed,
    # forward process is done by structure by structure manner.
    x = dict()

    if struct_weight:
        weight = item['struct_weight'].to(device=device, non_blocking=cuda)
    else:
        weight = torch.ones(n_batch).to(device=device, non_blocking=cuda)
    if inputs['neural_network']['use_force']:
        F = item['F'].type(dtype).to(device=device, non_blocking=cuda)
    if inputs['neural_network']['use_stress']:
        S = item['S'].type(dtype).to(device=device, non_blocking=cuda)


    E_ = 0.
    F_ = 0.
    S_ = 0.
    n_atoms = 0.
 
    #Loop for calculating energy
    for atype in inputs['atom_types']:
        x[atype] = item['x'][atype].to(device=device, non_blocking=cuda).requires_grad_(True)
        if x[atype].size(0) != 0:
            E_ += torch.sum(torch.sparse.DoubleTensor(
                item['sp_idx'][atype].long().to(device=device, non_blocking=cuda), 
                model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                item['sp_idx'][atype].size(1))).to_dense(), axis=1)
        n_atoms += item['n'][atype].to(device=device, non_blocking=cuda)
    #Energy loss + structure_weight
    e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=cuda) / n_atoms)
    print_e_loss = torch.mean(e_loss * weight)
    e_loss = torch.mean(e_loss * weight)

    #Loop for force, stress
    if inputs['neural_network']['use_force'] or inputs['neural_network']['use_stress']:
        #Loop for elements type
        for atype in inputs['atom_types']:
            if x[atype].size(0) != 0:
                dEdG = torch.autograd.grad(torch.sum(E_), x[atype], create_graph=True)[0]

                tmp_force = list()
                tmp_stress = list()
                tmp_idx = 0
                
                for n,ntem in enumerate(item['n'][atype]):
                    if inputs['neural_network']['use_force']: #force loop + structure_weight
                        if ntem != 0:
                            tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n].to(device=device, non_blocking=cuda), dEdG[tmp_idx:(tmp_idx + ntem)]))
                        else:
                            tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]).to(device=device, non_blocking=cuda))

                    if inputs['neural_network']['use_stress']: #stress loop
                        if ntem != 0:
                            tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n].to(device=device, non_blocking=cuda), dEdG[tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                        else:
                            tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]).to(device=device, non_blocking=cuda))
                    #Index sum
                    tmp_idx += ntem

                if inputs['neural_network']['use_force']:
                    F_ -= torch.cat(tmp_force, axis=0)

                if inputs['neural_network']['use_stress']:
                    S_ -= torch.cat(tmp_stress, axis=0) / units.GPa * 10

        #Force loss part 
        if inputs['neural_network']['use_force']:
            if inputs['neural_network']['force_diffscale']:
                # check the scale value: current = norm(force difference)
                # Force different scaling : larger force difference get higher weight !!
                force_diffscale = torch.sqrt(torch.norm(F_ - F, dim=1, keepdim=True).detach())
                f_loss = criterion(force_diffscale * F_, force_diffscale * F)
                print_f_loss = torch.mean(criterion(F_, F))
                #batch_idx = 0
                #for n in range(n_batch): #Make structure_weighted force
                #    tmp_idx = item['tot_num'][n].item()
                #    f_loss[batch_idx:(batch_idx+tmp_idx)] = f_loss[batch_idx:(batch_idx+tmp_idx)]*weight[n].item()
                #    batch_idx += tmp_idx 
                f_loss = torch.mean(f_loss)
            else:
                f_loss = criterion(F_, F)
                print_f_loss = torch.mean(f_loss)
                batch_idx = 0
                for n in range(n_batch): #Make structure_weighted force
                    tmp_idx = item['tot_num'][n].item()
                    f_loss[batch_idx:(batch_idx+tmp_idx)] = f_loss[batch_idx:(batch_idx+tmp_idx)] * weight[n].item()
                    batch_idx += tmp_idx 
                f_loss = torch.mean(f_loss)
            loss += inputs['neural_network']['force_coeff'] * f_loss
            progress_dict['f_err'].update(print_f_loss.detach().item(), F_.size(0))

        #Stress loss part
        if inputs['neural_network']['use_stress']:
            s_loss = criterion(S_, S)
            print_s_loss = torch.mean(s_loss)
            batch_idx = 0
            for n in range(n_batch): #Make structure_weighted force
                tmp_idx = item['tot_num'][n].item()
                s_loss[batch_idx:(batch_idx+tmp_idx)] = s_loss[batch_idx:(batch_idx+tmp_idx)] * weight[n].item()
                batch_idx += tmp_idx 
            s_loss = torch.mean(s_loss)
            loss += inputs['neural_network']['stress_coeff'] * s_loss
            progress_dict['s_err'].update(print_s_loss.detach().item(), n_batch)

    #Energy loss part
    loss = loss + inputs['neural_network']['energy_coeff'] * e_loss
    progress_dict['e_err'].update(print_e_loss.detach().item(), n_batch)
    loss = inputs['neural_network']['loss_scale'] * loss
    progress_dict['losses'].update(loss.detach().item(), n_batch)
    return loss



