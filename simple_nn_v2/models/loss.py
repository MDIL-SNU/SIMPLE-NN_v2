import torch
from ase import units

#Calculate E, F, S loss
def calculate_batch_loss(inputs, item, model, criterion, device, non_block, epoch_result, weighted, dtype, use_force, use_stress, atomic_e):
    n_batch = item['E'].size(0)
    weight = item['struct_weight'].to(device=device) if weighted else torch.ones(n_batch).to(device=device)
    calc_results = dict()

    x, atomic_E, E_, n_atoms = calculate_E(inputs['atom_types'], item, model, device, non_block)
    e_loss = get_e_loss(inputs['atom_types'], inputs['neural_network']['E_loss_type'], atomic_E, E_, n_atoms, item, criterion,\
                    epoch_result, dtype, device, non_block, n_batch, weight, atomic_e)
    batch_loss = inputs['neural_network']['energy_coeff'] * e_loss
    calc_results['E'] = E_

    if not atomic_e: # atomic_e training does not calculate F, S
        dEdG = calculate_derivative(inputs, inputs['atom_types'], x, E_)

        if use_force:
            F_ = calculate_F(inputs['atom_types'], x, dEdG, item, device, non_block)
            F = item['F'].type(dtype).to(device=device, non_blocking=non_block)
            if model.training:
                f_loss = get_f_loss(inputs['neural_network']['F_loss_type'], F_, F, criterion, epoch_result, n_batch, item, weight, gdf=inputs['neural_network']['atomic_weights'])
            else:
                f_loss = get_f_loss(inputs['neural_network']['F_loss_type'], F_, F, criterion, epoch_result, n_batch, item, weight, gdf=False)
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
                    tmp_force.append(torch.einsum('ijkl,ij->kl', item['dx'][atype][n].to(device=device, non_blocking=non_block), \
                    dEdG[atype][tmp_idx:(tmp_idx+ntem)]))
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
                    tmp_stress.append(torch.einsum('ijkl,ij->l', item['da'][atype][n].to(device=device, non_blocking=non_block), \
                    dEdG[atype][tmp_idx:(tmp_idx+ntem)]))
                else:
                    tmp_stress.append(torch.zeros(item['da'][atype][n].size()[-1]).to(device=device, non_blocking=non_block))
                tmp_idx += ntem

            S_ -= torch.cat(tmp_stress, axis=0) / units.GPa * 10

    return S_

def get_e_loss(atom_types, loss_type, atomic_E, E_, n_atoms, item, criterion, progress_dict, dtype, device, non_block, n_batch, weight, atomic_e):
    if not atomic_e: # Normal
        e_loss = 0.
        if loss_type == 1:
            e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=non_block) / n_atoms)
        elif loss_type == 2:
            e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=non_block) / n_atoms) * n_atoms
        else:
            e_loss = criterion(E_.squeeze(), item['E'].type(dtype).to(device=device, non_blocking=non_block))

        w_e_loss = torch.mean(e_loss * weight)
        for i, el in enumerate(e_loss):
            progress_dict['e_err'][item['struct_type'][i]].update(e_loss[i].detach().item())
        progress_dict['tot_e_err'].update(torch.mean(e_loss).detach(), n_batch)
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
                    for atom_idx in range(item['atomic_num'][atype][num].item()):
                        progress_dict['e_err'][item['struct_type'][num]].update(atype_loss[atype][tmp_idx+atom_idx].detach().item())
                    tmp_idx += item['atomic_num'][atype][num].item()
                print_e_loss.append(atype_loss[atype])
                atomic_loss.append(atype_loss[atype] * struct_weight_factor)

        w_e_loss = torch.mean(torch.cat(atomic_loss))
        print_e_loss = torch.mean(torch.cat(print_e_loss))
        progress_dict['tot_e_err'].update(print_e_loss.detach().item(), n_batch)

    return w_e_loss

def get_f_loss(loss_type, F_, F, criterion, progress_dict, n_batch, item, weight, gdf=False):
    if loss_type == 1:
        f_loss = criterion(F_, F)
    else:
        # check the scale value: current = norm(force difference)
        # Force different scaling : larger force difference get higher weight !!
        force_diffscale = torch.sqrt(torch.norm(F_ - F, dim=1, keepdim=True).detach())
        f_loss = criterion(force_diffscale * F_, force_diffscale * F)
    
    #GDF : using force weight scheme by G vector
    if gdf:
        batch_idx = 0
        for n in range(n_batch): #Make structure_weighted force
            tmp_idx = item['tot_num'][n].item()
            label = item['struct_type'][n]
            partial_f_loss = f_loss[batch_idx:batch_idx + tmp_idx]
            partial_gdf = item['atomic_weights'][batch_idx:batch_idx + tmp_idx]
            partial_f_mean = torch.mean(partial_f_loss)
            progress_dict['f_err'][label].update(partial_f_mean.detach().item() * 3, tmp_idx)
            progress_dict['tot_f_err'].update(partial_f_mean.detach().item() * 3, tmp_idx)
            partial_gdf = partial_gdf.view([-1, 1])
            partial_f_loss *= partial_gdf
            f_loss[batch_idx:batch_idx + tmp_idx] = partial_f_loss
            batch_idx += tmp_idx
    #Non GDF scheme. Use structure weight factor
    else:
        batch_idx = 0
        for n in range(n_batch): #Make structure_weighted force
            tmp_idx = item['tot_num'][n].item()
            label = item['struct_type'][n]
            partial_f_loss = f_loss[batch_idx:batch_idx + tmp_idx]
            partial_f_mean = torch.mean(partial_f_loss)
            progress_dict['f_err'][label].update(partial_f_mean.detach().item() * 3, tmp_idx)
            progress_dict['tot_f_err'].update(partial_f_mean.detach().item() * 3, tmp_idx)
            f_loss[batch_idx:batch_idx + tmp_idx] = partial_f_loss * weight[n].item() #* weight for gdf
            batch_idx += tmp_idx

    w_f_loss = torch.mean(f_loss)

    return w_f_loss

def get_s_loss(S_, S, criterion, progress_dict, n_batch, item, weight):
    s_loss = criterion(S_, S)
    batch_idx = 0
    for n in range(n_batch): #Make structure_weighted stress
        tmp_idx = 6
        label = item['struct_type'][n]
        partial_s_loss = torch.mean(s_loss[batch_idx:batch_idx + tmp_idx] * 6)
        progress_dict['s_err'][label].update(partial_s_loss.detach().item(), tmp_idx)
        progress_dict['tot_s_err'].update(partial_s_loss.detach().item(), tmp_idx)
        s_loss[batch_idx:batch_idx + tmp_idx] = s_loss[batch_idx:batch_idx + tmp_idx] * weight[n].item()
        batch_idx += tmp_idx

    w_s_loss = torch.mean(s_loss)

    return w_s_loss
