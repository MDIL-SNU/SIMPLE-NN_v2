import torch
import shutil
import time
from ase import units
from ..utils.Logger import AverageMeter, ProgressMeter


#This function train NN 
def train(inputs, logfile, data_loader, model, optimizer=None, criterion=None, scheduler=None, epoch=0, valid=False, save_result=False, cuda=False):

    ## Extract information of  use force & stress
    use_force = inputs['neural_network']['use_force']
    use_stress = inputs['neural_network']['use_stress']

    dtype = torch.get_default_dtype()

    progress, progress_dict = _init_meters(model, data_loader, optimizer, epoch, 
                                           valid, use_force, use_stress, save_result)

    end = time.time()
    max_len = len(data_loader)
    
    #Training part
    for i,item in enumerate(data_loader):
        progress_dict['data_time'].update(time.time() - end)

        loss = 0.
        e_loss = 0.
        n_batch = item['E'].size(0) + 1
        
        # Since the shape of input and intermediate state during forward is not fixed,
        # forward process is done by structure by structure manner.
        x = dict()
        
        if cuda and use_force: #GPU
            F = item['F'].type(dtype).cuda(non_blocking=True)
        elif use_force: #CPU
            F = item['F'].type(dtype)

        if cuda and use_stress: #GPU
            S = item['S'].type(dtype).cuda(non_blocking=True)
        elif use_stress: #CPU
            S = item['S'].type(dtype)
        
        E_ = 0.
        F_ = 0.
        S_ = 0.
        n_atoms = 0.
        
        #Loop
        if cuda: #GPU
            for atype in inputs['atom_types']:
                x[atype] = item['x'][atype].cuda(non_blocking=True).requires_grad_(True)
                if x[atype].size(0) != 0:
                    E_ += torch.sum(torch.sparse.DoubleTensor(
                        item['sp_idx'][atype].long().cuda(non_blocking=True), 
                        model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                        item['sp_idx'][atype].size(1))).to_dense(), axis=1)
                n_atoms += item['n'][atype].cuda(non_blocking=True)
        else: #CPU
            for atype in inputs['atom_types']:
                x[atype] = item['x'][atype].requires_grad_(True)
                if x[atype].size(0) != 0:
                    E_ += torch.sum(torch.sparse.DoubleTensor(
                        item['sp_idx'][atype].long(), 
                        model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0),
                        item['sp_idx'][atype].size(1))).to_dense(), axis=1)
                n_atoms += item['n'][atype]
           
        #LOSS
        if cuda: #GPU
            e_loss = criterion(E_.squeeze()/n_atoms, item['E'].type(dtype).cuda(non_blocking=True)/n_atoms)
        else: #CPU
            e_loss = criterion(E_.squeeze()/n_atoms, item['E'].type(dtype)/n_atoms)

       
        #Loop for force, stress
        if use_force or use_stress:
            #Loop for elements type
            for atype in inputs['atom_types']:
                if x[atype].size(0) != 0:
                    dEdG = torch.autograd.grad(torch.sum(E_), x[atype], create_graph=True)[0]

                    tmp_force = list()
                    tmp_stress = list()
                    tmp_idx = 0
                    
                    if cuda: #GPU
                        for n,ntem in enumerate(item['n'][atype]):
                            if use_force: #force loop
                                if ntem != 0:
                                    tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n].cuda(non_blocking=True), dEdG[tmp_idx:(tmp_idx + ntem)]))
                                else:
                                    tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]).cuda(non_blocking=True))

                            if use_stress: #stress loop
                                if ntem != 0:
                                    tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n].cuda(non_blocking=True), dEdG[tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                                else:
                                    tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]).cuda(non_blocking=True))
                            #Index sum
                            tmp_idx += ntem

                    else: #CPU
                        for n,ntem in enumerate(item['n'][atype]):
                            if use_force: #force loop
                                if ntem != 0:
                                    tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]))
                                else:
                                    tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]))

                            if use_stress: #stress loop
                                if ntem != 0:
                                    tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n], dEdG[tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                                else:
                                    tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]))
                            #Index sum
                            tmp_idx += ntem


                    if use_force:
                        F_ -= torch.cat(tmp_force, axis=0)

                    if use_stress:
                        S_ -= torch.cat(tmp_stress, axis=0) / units.GPa * 10

            #Force loss part 
            if use_force:
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
            if use_stress:
                s_loss = criterion(S_, S)
            
                loss += inputs['neural_network']['stress_coeff'] * s_loss
                progress_dict['s_err'].update(s_loss.detach().item(), n_batch)

        #Energy loss part
        loss = loss + inputs['neural_network']['energy_coeff'] * e_loss
        progress_dict['e_err'].update(e_loss.detach().item(), n_batch)
        progress_dict['losses'].update(loss.detach().item(), n_batch)
        loss = inputs['neural_network']['loss_scale'] * loss
        if not valid: #Back propagation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress_dict['batch_time'].update(time.time() - end)
        end = time.time()
        
        # max_len -> total size / batch size & i -> batch step in traning set
        # TODO: choose LOG file method
        if epoch % inputs['neural_network']['show_interval'] == 0 and not valid: #and i == max_len-1:
            progress.display(i+1)
            logfile.write(progress.log(i+1))
        elif epoch % inputs['neural_network']['show_interval'] == 0 and valid:
            progress.display(i+1)
            logfile.write(progress.log(i+1))


    return progress_dict['losses'].avg
    
def _init_meters(model, data_loader, optimizer, epoch, valid, use_force, use_stress, save_result):
    ## Setting LOG with progress meter
    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('loss', ':8.4e')
    e_err = AverageMeter('E err', ':6.4e', sqrt=True)
    progress_list = [losses, e_err]
    progress_dict = {'batch_time': batch_time, 'data_time': data_time, 'losses': losses, 'e_err': e_err} 
    
    if use_force:
        f_err = AverageMeter('F err', ':6.4e', sqrt=True)
        progress_list.append(f_err)
        progress_dict['f_err'] = f_err

    if use_stress:
        s_err = AverageMeter('S err', ':6.4e', sqrt=True)
        progress_list.append(s_err)
        progress_dict['s_err'] = s_err

    progress_list.append(batch_time)
    progress_list.append(data_time)

    if valid:
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
