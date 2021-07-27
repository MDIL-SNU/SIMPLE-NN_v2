import torch
from functools import partial
from braceexpand import braceexpand
from glob import glob

torch.set_default_dtype(torch.float64)

# Read specific parameter from pt file
class TorchStyleDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.data = torch.load(filename, map_location=torch.device('cpu'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#Open file contain directory of pytorch files and return them
class FilelistDataset(torch.utils.data.Dataset):
    def __init__(self, filename, device, load_data_to_gpu = False):
        if load_data_to_gpu:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.filelist = list()
        with open(filename) as fil:
            for line in fil:
                temp_list = glob(line.strip())
                temp_list.sort()
                for item in temp_list:
                    self.filelist.append(item)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        return torch.load(self.filelist[idx], map_location=self.device)
    
    def save_filename(self):
        for f in self.filelist:
            tmp_dict = torch.load(f, map_location=torch.device('cpu'))
            tmp_dict['filename'] = f
            torch.save(tmp_dict, f)

#Used in Structure rmse 
class StructlistDataset(FilelistDataset):
    def __init__(self, device, load_data_to_gpu = False):
        if load_data_to_gpu:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.filelist = list()

#Function to generate Iterator
def my_collate(batch, atom_types, device, scale_factor=None, pca=None, pca_min_whiten_level=None, use_stress=False, load_data_to_gpu=False):
    non_blocking = True if (load_data_to_gpu and torch.cuda.is_available()) else False 

    struct_weight = list()
    tot_num = list()
    E = list()
    x = _make_empty_dict(atom_types)
    n = _make_empty_dict(atom_types)

    F = list()
    dx = _make_empty_dict(atom_types)
    S = list()
    da = _make_empty_dict(atom_types)
    sparse_index = _make_empty_dict(atom_types)
    
    for item in batch:
        struct_weight.append(item['struct_weight'])
        tot_num.append(item['tot_num'])
        
        E.append(item['E'])
        for atype in atom_types: 
            x[atype].append(item['x'][atype])
            n[atype].append(item['x'][atype].size(0))
        
        F.append(item['F'])
        for atype in atom_types: 
            tmp_dx = item['dx'][atype] #Make dx to sparse_tensor
            if tmp_dx.is_sparse:
                tmp_dx = tmp_dx.to_dense().reshape(item['dx_size'][atype])
            tmp_dx = _preprocess_scaling_pca_to_dx_da(tmp_dx, atype, scale_factor, pca, pca_min_whiten_level)
            dx[atype].append(tmp_dx)

        if use_stress:
            S.append(item['S'])
            for atype in atom_types: 
                tmp_da = item['da'][atype]
                tmp_da = _preprocess_scaling_pca_to_dx_da(tmp_da, atype, scale_factor, pca, pca_min_whiten_level)
                da[atype].append(tmp_da)

    for atype in atom_types:
        x[atype] = torch.cat(x[atype], axis=0)
        x[atype] = _preprocess_scaling_and_pca_to_x(x, atype, scale_factor, pca, pca_min_whiten_level)
        n[atype] = _set_tensor_to_device(torch.tensor(n[atype]), device, non_blocking, load_data_to_gpu)
        sparse_index[atype] = _set_tensor_to_device(gen_sparse_index(n[atype]), device, non_blocking, load_data_to_gpu)

    struct_weight = _set_tensor_to_device(torch.tensor(struct_weight), device, non_blocking, load_data_to_gpu)
    tot_num = _set_tensor_to_device(torch.tensor(tot_num), device, non_blocking, load_data_to_gpu)
    E = _set_tensor_to_device(torch.tensor(E), device, non_blocking, load_data_to_gpu)
 
    F = torch.cat(F, axis=0)
    if use_stress:
        S = torch.cat(S, axis=0)

    return {'x': x, 'dx': dx, 'da': da, 'n': n, 'E': E, 'F': F, 'S': S, 'sp_idx': sparse_index, 'struct_weight': struct_weight, 'tot_num': tot_num}

#Function to generate Iterator
def atomic_e_collate(batch, atom_types, device, scale_factor=None, pca=None, pca_min_whiten_level=None, use_stress=False, load_data_to_gpu=False):
    non_blocking = True if (load_data_to_gpu and torch.cuda.is_available()) else False 

    struct_weight = list()
    tot_num = list()
    E = list()
    x = _make_empty_dict(atom_types)
    n = _make_empty_dict(atom_types)

    atomic_E = _make_empty_dict(atom_types)
    atomic_num = _make_empty_dict(atom_types)
    sparse_index = _make_empty_dict(atom_types)

    # extract data from batch_loader
    for item in batch:
        struct_weight.append(item['struct_weight'])
        tot_num.append(item['tot_num'])

        E.append(item['E'])
        for atype in atom_types: 
            x[atype].append(item['x'][atype])
            n[atype].append(item['x'][atype].size(0))

        for atype in atom_types: 
            if atype in item['atomic_E'].keys():
                atomic_E[atype].append(item['atomic_E'][atype])
                atomic_num[atype].append(item['atomic_E'][atype].size(0))
            else:
                atomic_num[atype].append(0)

    # load data to torch tensor
    for atype in atom_types:
        x[atype] = torch.cat(x[atype], axis=0)
        x[atype] = _preprocess_scaling_and_pca_to_x(x, atype, scale_factor, pca, pca_min_whiten_level)
        n[atype] = _set_tensor_to_device(torch.tensor(n[atype]), device, non_blocking, load_data_to_gpu)
        sparse_index[atype] = _set_tensor_to_device(gen_sparse_index(n[atype]), device, non_blocking, load_data_to_gpu)

        atomic_E[atype] = torch.cat(atomic_E[atype]) if atomic_E[atype] else None
        atomic_num[atype] = _set_tensor_to_device(torch.tensor(atomic_num[atype]), device, non_blocking, load_data_to_gpu)
    
    struct_weight = _set_tensor_to_device(torch.tensor(struct_weight), device, non_blocking, load_data_to_gpu)
    tot_num = _set_tensor_to_device(torch.tensor(tot_num), device, non_blocking, load_data_to_gpu)
    E = _set_tensor_to_device(torch.tensor(E), device, non_blocking, load_data_to_gpu)

    return {'x': x, 'n': n, 'E': E, 'atomic_E': atomic_E, 'sp_idx': sparse_index, 'struct_weight': struct_weight, 'tot_num': tot_num, 'atomic_num': atomic_num}

def _make_empty_dict(atom_types):
    dic = dict()
    for atype in atom_types:
        dic[atype] = list()

    return dic

def _preprocess_scaling_and_pca_to_x(x, atype, scale_factor, pca, pca_min_whiten_level):
    if scale_factor: #Scale part
        x[atype] -= scale_factor[atype][0].view(1,-1)
        x[atype] /= scale_factor[atype][1].view(1,-1)
    if pca: #PCA part
        if x[atype].size(0) != 0:
            x[atype] = torch.einsum('ij,jm->im', x[atype], pca[atype][0]) - pca[atype][2].reshape(1,-1)
        if pca_min_whiten_level:
            x[atype] /= pca[atype][1].view(1,-1)

    return x[atype]

def _preprocess_scaling_pca_to_dx_da(tmp_dx, atype, scale_factor, pca, pca_min_whiten_level):
    if scale_factor:
        tmp_dx /= scale_factor[atype][1].view(1,-1,1,1)
    if pca:
        if tmp_dx.size(0) != 0:
            tmp_dx = torch.einsum('ijkl,jm->imkl', tmp_dx, pca[atype][0])
        if pca_min_whiten_level:
            tmp_dx /= pca[atype][1].view(1,-1,1,1)
    return tmp_dx

def _set_tensor_to_device(tensor, device, non_blocking, load_data_to_gpu):
    if load_data_to_gpu:
        tensor.to(device=device, non_blocking=non_blocking)

    return tensor

#Function to generate Iterator
def filename_collate(batch, atom_types, device, scale_factor=None, pca=None, pca_min_whiten_level=None, use_stress=False, load_data_to_gpu=False):
    tmp_dict = my_collate(batch, atom_types, device, scale_factor, pca, pca_min_whiten_level, use_stress, load_data_to_gpu)
    tmp_list = list()
    for item in batch:
        tmp_list.append(item['filename'])
    tmp_dict['filename'] = tmp_list
    return tmp_dict

def gen_sparse_index(nlist):
    res = torch.zeros(2, sum(nlist))
    idx = 0
    for i,item in enumerate(nlist):
        for jtem in range(item):
            res[0, idx] = i
            res[1, idx] = idx
            idx += 1
    return res

def delete_key_in_pt(filename, key):       
    filelist = FilelistDataset(filename)
    filelist.save_filename()  
    for f in range(filelist):
        pt_dict = f
        del pt_dict['filename']
        if key in pt_dict.keys():
            del pt_dict[key]
        torch.save(pt_dict, f['filename'])

def _load_dataset(inputs, logfile, scale_factor, pca, device, mode):
    # mode: ['train', 'valid', 'test', 'add_NNP_ref', 'atomic_E_train', 'atomic_E_valid']
    args = {
        'train': {'data_list': inputs['descriptor']['train_list'], 'use_stress': inputs['neural_network']['use_stress'], 'valid': False, 'my_collate': my_collate},
        'valid': {'data_list': inputs['descriptor']['valid_list'], 'use_stress': inputs['neural_network']['use_stress'], 'valid': True, 'my_collate': my_collate},
        'test': {'data_list': inputs['descriptor']['test_list'], 'use_stress': inputs['neural_network']['use_stress'], 'valid': True, 'my_collate': my_collate},
        'add_NNP_ref': {'data_list': inputs['descriptor']['ref_list'], 'use_stress': False, 'valid': True, 'my_collate': filename_collate},
        'atomic_E_train': {'data_list': inputs['descriptor']['train_list'], 'use_stress': False, 'valid': False, 'my_collate': atomic_e_collate},
        'atomic_E_valid': {'data_list': inputs['descriptor']['train_list'], 'use_stress': False, 'valid': True, 'my_collate': atomic_e_collate}
    }

    dataset_list = FilelistDataset(args[mode]['data_list'], device, inputs['neural_network']['load_data_to_gpu'])
    if mode == 'add_NNP_ref':
        dataset_list.save_filename()
    data_loader = _make_dataloader(inputs, dataset_list, scale_factor, pca, device, args[mode]['use_stress'], args[mode]['valid'], args[mode]['my_collate'])
    logfile.write("{} dataset loaded\n".format(mode))

    return data_loader

def _load_labeled_dataset(inputs, logfile, scale_factor, pca, device, mode):
    # mode: ['train', 'valid', 'test', 'atomic_E_train', 'atomic_E_valid']
    args = {
        'train': {'data_list': inputs['descriptor']['train_list'], 'use_stress': inputs['neural_network']['use_stress'], 'valid': False, 'my_collate': my_collate},
        'valid': {'data_list': inputs['descriptor']['valid_list'], 'use_stress': inputs['neural_network']['use_stress'], 'valid': True, 'my_collate': my_collate},
        'test': {'data_list': inputs['descriptor']['test_list'], 'use_stress': inputs['neural_network']['use_stress'], 'valid': True, 'my_collate': my_collate},
        'atomic_E_train': {'data_list': inputs['descriptor']['train_list'], 'use_stress': False, 'valid': False, 'my_collate': atomic_e_collate},
        'atomic_E_valid': {'data_list': inputs['descriptor']['valid_list'], 'use_stress': False, 'valid': True, 'my_collate': atomic_e_collate}
    }
    
    labeled_dataset_dict = _set_struct_dict(args[mode]['data_list'], device, inputs['neural_network']['load_data_to_gpu'])
    labeled_data_loader = dict()

    for key in labeled_dataset_dict.keys():
        labeled_data_loader[key] = _make_dataloader(inputs, labeled_dataset_dict[key], scale_factor, pca, device, args[mode]['use_stress'], args[mode]['valid'], args[mode]['my_collate'])
    logfile.write("labeled {} dataset loaded\n".format(mode))

    return labeled_data_loader

#Load collate from train, valid dataset
def _make_dataloader(inputs, dataset_list, scale_factor, pca, device, use_stress, valid, my_collate=my_collate):
    if len(dataset_list) == 0:
        data_loader = None
    else:
        batch_size = len(dataset_list) if inputs['neural_network']['full_batch'] else inputs['neural_network']['batch_size']
        shuffle = False if valid else inputs['descriptor']['shuffle']

        partial_collate = partial(my_collate, atom_types=inputs['atom_types'], device=device,
            scale_factor=scale_factor, pca=pca, pca_min_whiten_level=inputs['neural_network']['pca_min_whiten_level'],
            use_stress=use_stress, load_data_to_gpu=inputs['neural_network']['load_data_to_gpu'])

        data_loader = torch.utils.data.DataLoader(
            dataset_list, batch_size=batch_size, shuffle=shuffle, collate_fn=partial_collate,
            num_workers=inputs['neural_network']['workers'], pin_memory=False)

    return data_loader

#Function that set structure 
def _set_struct_dict(filename, device, load_data_to_gpu=False):
    structure_dict = dict()
    with open(filename) as fil:
        for line in fil:
            temp_list = glob(line.strip())
            temp_list.sort()
            ## Weight check and copy
            for item in temp_list:
                tmp_name = TorchStyleDataset(item)['struct_type']
                if not tmp_name in structure_dict.keys():
                    structure_dict[tmp_name] = StructlistDataset(device, load_data_to_gpu=load_data_to_gpu)
                structure_dict[tmp_name].filelist.append(item)

    return structure_dict
