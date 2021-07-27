import torch
import time
from ase import units
from simple_nn_v2.models.logger import AverageMeter, TimeMeter


def _init_meters(use_force, use_stress, atomic_e):
    ## Setting LOG with progress meter
    losses = AverageMeter('loss', ':8.4e')
    e_err = AverageMeter('E err', ':6.4e', sqrt=True)
    batch_time = TimeMeter('time', ':6.3f')
    data_time = TimeMeter('data', ':6.3f')
    total_time = TimeMeter('total time', ':8.4e')
    progress_dict = {'batch_time': batch_time, 'data_time': data_time, 'losses': losses, 'e_err': e_err, 'total_time': total_time}

    if use_force and not atomic_e:
        f_err = AverageMeter('F err', ':6.4e', sqrt=True)
        progress_dict['f_err'] = f_err
    if use_stress and not atomic_e:
        s_err = AverageMeter('S err', ':6.4e', sqrt=True)
        progress_dict['s_err'] = s_err

    return progress_dict

def calculate_batch_loss(inputs, item, model, criterion, device, non_block, epoch_result, weighted, dtype, use_force, use_stress, atomic_e):
    n_batch = item['E'].size(0) 
    weight = item['struct_weight'] if weighted else torch.ones(n_batch, device=device)
    weight.to(device=device, non_blocking=non_block)
    calc_results = dict()

    x, atomic_E, E_, n_atoms = calculate_E(inputs['atom_types'], item, model, device, non_block)
    e_loss = get_e_loss(inputs['atom_types'], inputs['neural_network']['E_loss_type'], atomic_E, E_, n_atoms, item, criterion, epoch_result, dtype, device, non_block, n_batch, weight, atomic_e)
    batch_loss = inputs['neural_network']['energy_coeff'] * e_loss
    calc_results['E'] = E_

    if not atomic_e: # atomic_e training does not calculate F, S
        dEdG = calculate_derivative(inputs, inputs['atom_types'], x, E_)
        if use_force:
            F_ = calculate_F(inputs['atom_types'], x, dEdG, item, device, non_block)
            F = item['F'].type(dtype).to(device=device, non_blocking=non_block)
            f_loss = get_f_loss(inputs['neural_network']['F_loss_type'], F_, F, criterion, epoch_result, n_batch, item, weight)
            batch_loss += inputs['neural_network']['force_coeff'] * f_loss
            calc_results['F'] = F_

        if use_stress:
            S_ = calculate_S(inputs['atom_types'], x, dEdG, item, device, non_block)
            S = item['S'].type(dtype).to(device=device, non_blocking=non_block)
            s_loss = get_s_loss(S_, S, criterion, epoch_result, n_batch, item, weight)
            batch_loss += inputs['neural_network']['stress_coeff'] * s_loss
            calc_results['S'] = S_

    batch_loss *= inputs['neural_network']['loss_scale']
    epoch_result['losses'].update(batch_loss.detach().item(), n_batch)

    return batch_loss, calc_results

def calculate_E(atom_types, item, model, device, non_block):
    x = dict()
    atomic_E = dict()
    E_ = 0
    n_atoms = 0

    for atype in atom_types:
        x[atype] = item['x'][atype].to(device=device, non_blocking=non_block).requires_grad_(True)
        atomic_E[atype] = None
        if x[atype].size(0) != 0:
            atomic_E[atype] = torch.sparse.DoubleTensor(
                item['sp_idx'][atype].long().to(device=device, non_blocking=non_block), model.nets[atype](x[atype]).squeeze(),
                size=(item['n'][atype].size(0), item['sp_idx'][atype].size(1))
            ).to_dense()
            E_ += torch.sum(atomic_E[atype], axis=1)
        n_atoms += item['n'][atype].to(device=device, non_blocking=non_block)
    
    return x, atomic_E, E_, n_atoms

def calculate_derivative(inputs, atom_types, x, E_):
    dEdG = dict()
    if inputs['neural_network']['use_force'] or inputs['neural_network']['use_stress']:
        for atype in atom_types:
            if x[atype].size(0) != 0:
                dEdG[atype] = torch.autograd.grad(torch.sum(E_), x[atype], create_graph=True)[0]

    return dEdG
    
def calculate_F(atom_types, x, dEdG, item, device, non_block):
    F_ = 0.
    for atype in atom_types:
        if x[atype].size(0) != 0:
            tmp_force = list()
            tmp_idx = 0

            for n, ntem in enumerate(item['n'][atype]):
                if ntem != 0:
                    tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n].to(device=device, non_blocking=non_block), dEdG[atype][tmp_idx:(tmp_idx + ntem)]))
                else:
                    tmp_force.append(torch.zeros(item['dx'][atype][n].size()[-2], item['dx'][atype][n].size()[-1]).to(device=device, non_blocking=non_block))
                tmp_idx += ntem

            F_ -= torch.cat(tmp_force, axis=0)

    return F_

def calculate_S(atom_types, x, dEdG, item, device, non_block):
    S_ = 0.
    for atype in atom_types:
        if x[atype].size(0) != 0:
            tmp_stress = list()
            tmp_idx = 0

            for n, ntem in enumerate(item['n'][atype]):
                if ntem != 0:
                    tmp_stress.append(torch.einsum('ijkl,ij->kl', item['da'][atype][n].to(device=device, non_blocking=non_block), dEdG[atype][tmp_idx:(tmp_idx + ntem)]).sum(axis=0))
                else:
                    tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]).to(device=device, non_blocking=non_block))

            S_ -= torch.cat(tmp_stress, axis=0) / units.GPa * 10

    return S_

def get_e_loss(atom_types, loss_type, atomic_E, E_, n_atoms, item, criterion, progress_dict, dtype, device, non_block, n_batch, weight, atomic_e):
    if not atomic_e: # Normal
        e_loss = 0.
        if loss_type == 1:
            e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=non_block) / n_atoms) * n_atoms
        elif loss_type == 2:
            e_loss = criterion(E_.squeeze(), item['E'].type(dtype).to(device=device, non_blocking=non_block))
        else:
            e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=non_block) / n_atoms)

        w_e_loss = torch.mean(e_loss * weight)
        e_loss = torch.mean(e_loss)
        progress_dict['e_err'].update(e_loss.detach().item(), n_batch)
    else: # atomic E using at replica train
        atype_loss = dict()
        for atype in atom_types:
            if atomic_E[atype] is not None:
                batch_sum = torch.sum(atomic_E[atype], axis=0) 
                atype_loss[atype] = criterion(batch_sum, item['atomic_E'][atype].type(dtype).to(device=device, non_blocking=non_block))
            else:
                atype_loss[atype] = None

        struct_weight_factor = None
        print_e_loss = list()
        atomic_loss = list()
        for atype in atom_types:
            if atype_loss[atype] is not None:
                struct_weight_factor = torch.zeros(torch.sum(item['atomic_num'][atype]), device=device)
                tmp_idx = 0
                for num in range(n_batch):
                    struct_weight_factor[tmp_idx:tmp_idx+item['atomic_num'][atype][num].item()] += weight[num]
                    tmp_idx += item['atomic_num'][atype][num].item()
                print_e_loss.append(atype_loss[atype])
                atomic_loss.append(atype_loss[atype]*struct_weight_factor)

        w_e_loss = torch.mean(torch.cat(atomic_loss))
        print_e_loss = torch.mean(torch.cat(print_e_loss))
        progress_dict['e_err'].update(print_e_loss.detach().item(), n_batch)

    return w_e_loss

def get_f_loss(loss_type, F_, F, criterion, progress_dict, n_batch, item, weight):
    if loss_type == 2:
        # check the scale value: current = norm(force difference)
        # Force different scaling : larger force difference get higher weight !!
        force_diffscale = torch.sqrt(torch.norm(F_ - F, dim=1, keepdim=True).detach())
        f_loss = criterion(force_diffscale * F_, force_diffscale * F)
        #aw_factor need
    else:
        f_loss = criterion(F_, F)
        batch_idx = 0
        for n in range(n_batch): #Make structure_weighted force
            tmp_idx = item['tot_num'][n].item()
            f_loss[batch_idx:(batch_idx+tmp_idx)] = f_loss[batch_idx:(batch_idx+tmp_idx)] * weight[n].item()
            batch_idx += tmp_idx

    w_f_loss = torch.mean(f_loss)
    print_f_loss = torch.mean(criterion(F_, F))
    progress_dict['f_err'].update(print_f_loss.detach().item(), F_.size(0))

    return w_f_loss

def get_s_loss(S_, S, criterion, progress_dict, n_batch, item, weight):
    s_loss = criterion(S_, S)
    batch_idx = 0
    for n in range(n_batch): #Make structure_weighted force
        tmp_idx = item['tot_num'][n].item()
        s_loss[batch_idx:(batch_idx+tmp_idx)] = s_loss[batch_idx:(batch_idx+tmp_idx)] * weight[n].item()
        batch_idx += tmp_idx

    w_s_loss = torch.mean(s_loss)
    s_loss = torch.mean(criterion(S_, S))
    progress_dict['s_err'].update(s_loss.detach().item(), n_batch)

    return w_s_loss
