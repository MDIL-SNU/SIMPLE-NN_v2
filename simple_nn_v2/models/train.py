import torch
import shutil
import time
from ase import units
from ..utils.Logger import AverageMeter, ProgressMeter

def train(data_loader, model, optimizer=None, criterion=None, scheduler=None, epoch=0, valid=False, save_result=False, inputs=None, cuda=False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    e_err = AverageMeter('E err', ':6.4e', sqrt=True)
    progress_list = [batch_time, data_time, losses, e_err]
    
    if inputs['neural_network']['use_force']:
        f_err = AverageMeter('F err', ':6.4e', sqrt=True)
        progress_list.append(f_err)

    if inputs['neural_network']['use_stress']:
        s_err = AverageMeter('S err', ':6.4e', sqrt=True)
        progress_list.append(s_err)

    dtype = torch.get_default_dtype()

    if valid:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix=f"Valid (GPU:{0}): [{epoch}]",
            #suffix=f"lr: {optimizer.param_groups[0]['lr']:6.4e}"
        )

        model.eval()
    else:
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix=f"Epoch (GPU:{0}): [{epoch}]",
            suffix=f"lr: {optimizer.param_groups[0]['lr']:6.4e}"
        )
        
        model.train()

    if save_result:
        res_dict = {
            'NNP_E': list(),
            'NNP_F': list(),
            'DFT_E': list(),
            'DFT_F': list(),
        }

    end = time.time()

    max_len = len(data_loader)
    

    
    for i,item in enumerate(data_loader):
        data_time.update(time.time() - end)

        loss = 0.
        e_loss = 0.
        n_batch = item['E'].size(0) + 1
        
        # Since the shape of input and intermediate state during forward is not fixed,
        # forward process is done by structure by structure manner.
        x = dict()
        
        if cuda and inputs['neural_network']['use_force']: #GPU
            F = item['F'].type(dtype).cuda(non_blocking=True)
        elif inputs['neural_network']['use_force']: #CPU
            F = item['F'].type(dtype)

        if cuda and inputs['neural_network']['use_stress']: #GPU
            S = item['S'].type(dtype).cuda(non_blocking=True)
        elif inputs['neural_network']['use_stress']: #CPU
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
                        model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0), item['sp_idx'][atype].size(1))).to_dense(), axis=1)
                n_atoms += item['n'][atype].cuda(non_blocking=True)
        else: #CPU
            for atype in inputs['atom_types']:
                x[atype] = item['x'][atype].requires_grad_(True)
                if x[atype].size(0) != 0:
                    E_ += torch.sum(torch.sparse.DoubleTensor(
                        item['sp_idx'][atype].long(), 
                        model.nets[atype](x[atype]).squeeze(), size=(item['n'][atype].size(0), item['sp_idx'][atype].size(1))).to_dense(), axis=1)
                n_atoms += item['n'][atype]
    
        
        #LOSS
        if cuda: #GPU
            e_loss = criterion(E_.squeeze()/n_atoms, item['E'].type(dtype).cuda(non_blocking=True)/n_atoms)
        else: #CPU
            e_loss = criterion(E_.squeeze()/n_atoms, item['E'].type(dtype)/n_atoms)

       
        #Loop for force, stress
        if inputs['neural_network']['use_force'] or inputs['neural_network']['use_stress']:
            for atype in inputs['atom_types']:
                if x[atype].size(0) != 0:
                    dEdG = torch.autograd.grad(torch.sum(E_), x[atype], create_graph=True)[0]

                    tmp_force = list()
                    tmp_stress = list()
                    tmp_idx = 0
                    

                    #Loop for type
                    if cuda: #GPU
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

                    else: #CPU
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
            
            if inputs['neural_network']['use_force']:
                if inputs['neural_network']['force_diffscale']:
                    # check the scale value: current = norm(force difference)
                    force_diffscale = torch.sqrt(torch.norm(F_ - F, dim=1, keepdim=True).detach())

                    f_loss = criterion(force_diffscale * F_, force_diffscale * F)
                    print_f_loss = criterion(F_, F)
                else:
                    f_loss = criterion(F_, F)
                    print_f_loss = f_loss

                loss += inputs['neural_network']['force_coeff'] * f_loss
                f_err.update(print_f_loss.detach().item(), F_.size(0))

            if inputs['neural_network']['use_stress']:
                s_loss = criterion(S_, S)
            
                loss += inputs['neural_network']['stress_coeff'] * s_loss
                s_err.update(s_loss.detach().item(), n_batch)

        loss = loss + inputs['neural_network']['energy_coeff'] * e_loss

        e_err.update(e_loss.detach().item(), n_batch)
        losses.update(loss.detach().item(), n_batch)

        if not valid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % inputs['neural_network']['show_interval'] == 0 or i == max_len-1:
            progress.display(i)

    return losses.avg
    

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
