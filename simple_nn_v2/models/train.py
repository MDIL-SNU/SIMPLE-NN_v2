import torch
import shutil
import time
from ase import units
from ..utils.Logger import AverageMeter, ProgressMeter, TimeMeter, StructureMeter


#This function train NN 
def train(inputs, logfile, data_loader, model, optimizer=None, criterion=None, scheduler=None, epoch=0, valid=False, save_result=False, cuda=False, err_dict=None,start_time=None, test=False):

    dtype = torch.get_default_dtype()

    progress, progress_dict = _init_meters(model, data_loader, optimizer, epoch, 
                                           valid, inputs['neural_network']['use_force'], inputs['neural_network']['use_stress'], save_result, test)

    end = time.time()
    max_len = len(data_loader)
    
    #Training part
    for i,item in enumerate(data_loader):
        progress_dict['data_time'].update(time.time() - end)
        
        if cuda:
            loss = _loop_for_gpu(inputs, item, dtype, model, criterion, progress_dict)
        else:
            loss = _loop_for_cpu(inputs, item, dtype, model, criterion, progress_dict)

        if not valid and not test: #Back propagation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress_dict['batch_time'].update(time.time() - end)
        end = time.time()
        
        # max_len -> total size / batch size & i -> batch step in traning set
        # TODO: choose LOG file method
        if test:
            progress.test()
        elif epoch % inputs['neural_network']['show_interval'] == 0 and i == max_len-1:
            progress_dict['total_time'].update(time.time() - start_time)
            progress.display(i+1)
            logfile.write(progress.log(i+1))
    
    # After one epoch load err_dict & use_force, use_stress included err_dict
    if err_dict:
        for err_type in err_dict.keys():
            err_dict[err_type][0] = progress_dict[err_type].avg

    return progress_dict['losses'].avg
    
def _init_meters(model, data_loader, optimizer, epoch, valid, use_force, use_stress, save_result, test):
    ## Setting LOG with progress meter
    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('loss', ':8.4e')
    e_err = AverageMeter('E err', ':6.4e', sqrt=True)
    total_time = TimeMeter('total time',':8.4e')
    progress_list = [losses, e_err]
    progress_dict = {'batch_time': batch_time, 'data_time': data_time, 'losses': losses, 'e_err': e_err, 'total_time':total_time} 
    
    if use_force:
        f_err = AverageMeter('F err', ':6.4e', sqrt=True)
        progress_list.append(f_err)
        progress_dict['f_err'] = f_err

    if use_stress:
        s_err = AverageMeter('S err', ':6.4e', sqrt=True)
        progress_list.append(s_err)
        progress_dict['s_err'] = s_err
    
    if not test:
        progress_list.append(batch_time)
        progress_list.append(data_time)
        progress_list.append(total_time)

    if test:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Evaluation : ",
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
    #Not used yet
    '''
    if save_result:
        res_dict = {
            'NNP_E': list(),
            'NNP_F': list(),
            'DFT_E': list(),
            'DFT_F': list(),
        }
    '''
    return progress, progress_dict

#Show structure rmse
def _show_structure_rmse(inputs, logfile, train_struct_dict, valid_struct_dict, model, optimizer=None, criterion=None, cuda=False, test=False):
    for t_key in train_struct_dict.keys():
        log_train = _struct_log(inputs, train_struct_dict[t_key], model, optimizer=optimizer, criterion=criterion, cuda=cuda, test=test)
        log_train = "[{0:8}] ".format(t_key)+log_train

        if valid_struct_dict[t_key]:
            log_valid = _struct_log(inputs, valid_struct_dict[t_key], model,valid=True, optimizer=optimizer, criterion=criterion, cuda=cuda)
            log_valid = "\n[{0:8}] ".format(t_key)+log_valid

        else:
            log_valid = ""
        outdict = log_train+log_valid

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
        f_err = StructureMeter('F err', ':6.4e',sqrt=True)
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
        if cuda:
            loss = _loop_for_gpu(inputs, item, dtype, model, criterion, progress_dict)
        else:
            loss = _loop_for_cpu(inputs, item, dtype, model, criterion, progress_dict)
    
    return progress.string()

#Traning loop for CPU 
def _loop_for_cpu(inputs, item, dtype, model, criterion, progress_dict):
    loss = 0.
    e_loss = 0.
    n_batch = item['E'].size(0) + 1
    
    # Since the shape of input and intermediate state during forward is not fixed,
    # forward process is done by structure by structure manner.
    x = dict()

    if inputs['neural_network']['use_force']:
        F = item['F'].type(dtype)
    if inputs['neural_network']['use_stress']:
        S = item['S'].type(dtype)
    
    E_ = 0.
    F_ = 0.
    S_ = 0.
    n_atoms = 0.
 
    #Loop
    for atype in inputs['atom_types']:
        x[atype] = item['x'][atype].requires_grad_(True)
        if x[atype].size(0) != 0:
            E_ += torch.sum(torch.sparse.DoubleTensor(
                item['sp_idx'][atype].long(), 
                model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                item['sp_idx'][atype].size(1))).to_dense(), axis=1)
        n_atoms += item['n'][atype]
       
    #LOSS
    e_loss = criterion(E_.squeeze()/n_atoms, item['E'].type(dtype)/n_atoms)

   
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
                    if inputs['neural_network']['use_force']: #force loop
                        if ntem != 0:
                            tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]))
                        else:
                            tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]))

                    if inputs['neural_network']['use_stress']: #stress loop
                        if ntem != 0:
                            tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                        else:
                            tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]))
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
                print_f_loss = criterion(F_, F)
            else:
                f_loss = criterion(F_, F)
                print_f_loss = f_loss

            loss += inputs['neural_network']['force_coeff'] * f_loss
            progress_dict['f_err'].update(print_f_loss.detach().item(), F_.size(0))
        #Stress loss part
        if inputs['neural_network']['use_stress']:
            s_loss = criterion(S_, S)
        
            loss += inputs['neural_network']['stress_coeff'] * s_loss
            progress_dict['s_err'].update(s_loss.detach().item(), n_batch)

    #Energy loss part
    loss = loss + inputs['neural_network']['energy_coeff'] * e_loss
    progress_dict['e_err'].update(e_loss.detach().item(), n_batch)
    progress_dict['losses'].update(loss.detach().item(), n_batch)
    loss = inputs['neural_network']['loss_scale'] * loss

    return loss

#Traning loop for GPU 
def _loop_for_gpu(inputs, item, dtype, model, criterion, progress_dict):
    loss = 0.
    e_loss = 0.
    n_batch = item['E'].size(0) + 1
    
    # Since the shape of input and intermediate state during forward is not fixed,
    # forward process is done by structure by structure manner.
    x = dict()

    if inputs['neural_network']['use_force']:
        F = item['F'].type(dtype).cuda(non_blocking=True)
    if inputs['neural_network']['use_stress']:
        S = item['S'].type(dtype).cuda(non_blocking=True)
    
    E_ = 0.
    F_ = 0.
    S_ = 0.
    n_atoms = 0.
 
    #Loop
    for atype in inputs['atom_types']:
        x[atype] = item['x'][atype].cuda(non_blocking=True).requires_grad_(True)
        if x[atype].size(0) != 0:
            E_ += torch.sum(torch.sparse.DoubleTensor(
                item['sp_idx'][atype].long().cuda(non_blocking=True), 
                model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                item['sp_idx'][atype].size(1))).to_dense(), axis=1)
        n_atoms += item['n'][atype].cuda(non_blocking=True)
    #Energy loss
    e_loss = criterion(E_.squeeze()/n_atoms, item['E'].type(dtype).cuda(non_blocking=True)/n_atoms)
   
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
                    if inputs['neural_network']['use_force']: #force loop
                        if ntem != 0:
                            tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n].cuda(non_blocking=True), dEdG[tmp_idx:(tmp_idx + ntem)]))
                        else:
                            tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]).cuda(non_blocking=True))

                    if inputs['neural_network']['use_stress']: #stress loop
                        if ntem != 0:
                            tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n].cuda(non_blocking=True), dEdG[tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                        else:
                            tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]).cuda(non_blocking=True))
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
                print_f_loss = criterion(F_, F)
            else:
                f_loss = criterion(F_, F)
                print_f_loss = f_loss
            loss += inputs['neural_network']['force_coeff'] * f_loss
            progress_dict['f_err'].update(print_f_loss.detach().item(), F_.size(0))

        #Stress loss part
        if inputs['neural_network']['use_stress']:
            s_loss = criterion(S_, S)
            loss += inputs['neural_network']['stress_coeff'] * s_loss
            progress_dict['s_err'].update(s_loss.detach().item(), n_batch)

    #Energy loss part
    loss = loss + inputs['neural_network']['energy_coeff'] * e_loss
    progress_dict['e_err'].update(e_loss.detach().item(), n_batch)
    progress_dict['losses'].update(loss.detach().item(), n_batch)
    loss = inputs['neural_network']['loss_scale'] * loss
  
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

