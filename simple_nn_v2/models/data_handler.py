import torch
from functools import partial
from braceexpand import braceexpand
from glob import glob
from six.moves import cPickle as pickle

torch.set_default_dtype(torch.float64)

# From pickle files
class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, filename, atom_types):
        with open(filename) as fil:
            self.files = [line.strip() for line in fil]
        self.atom_types = atom_types

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as fil:
            datum = pickle.load(fil)
        x = dict()
        dx = dict()
        for atype in self.atom_types:
            x[atype] = torch.tensor(datum['x'][atype])
            dx[atype] = torch.tensor(datum['dx'][atype])
        E = datum['E']
        F = torch.tensor(datum['F'])

        return {'x': x, 'dx': dx, 'E': E, 'F': F}

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
    def __init__(self, filename):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return torch.load(self.filelist[idx], map_location=torch.device('cpu'))
        #return torch.load(self.filelist[idx], map_location=self.device)
    
    def save_filename(self):
        for f in self.filelist:
            tmp_dict = torch.load(f, map_location=torch.device('cpu'))
            tmp_dict['filename'] = f
            torch.save(tmp_dict, f)

#Used in Structure rmse 
class StructlistDataset(FilelistDataset):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filelist = list()


#Function that set structure 
def _set_struct_dict(filename):
    structure_dict = dict()
    with open(filename) as fil:
        try: #Check valid set exist
            print(fil[0])
            valid = True
        except:
            valid = False 

        for line in fil:
            temp_list = glob(line.strip())
            temp_list.sort()
            ## Weight check and copy
            for item in temp_list:
                tmp_name = TorchStyleDataset(item)['struct_type']
                if not tmp_name in structure_dict.keys():
                    structure_dict[tmp_name] =  StructlistDataset()
                structure_dict[tmp_name].filelist.append(item)

    return structure_dict

#Function to generate Iterator
def my_collate(batch, atom_types, scale_factor=None, pca=None, pca_min_whiten_level=None, use_stress=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    non_block = True
    x = dict()
    dx = dict()
    da = dict()
    n = dict()
    sparse_index = dict()
    for atype in atom_types:
        x[atype] = list()
        dx[atype] = list()
        da[atype] = list()
        n[atype] = list()
    E = list()
    F = list()
    S = list()
    struct_weight = list()
    tot_num = list()
    '''
    for atype in atom_types:
        if scale_factor:
            scale_factor[atype][0] = scale_factor[atype][0].to(device=device, non_blocking=non_block)
            scale_factor[atype][1] = scale_factor[atype][1].to(device=device, non_blocking=non_block)
        if pca:
            pca[atype][0] = pca[atype][0].to(device=device, non_blocking=non_block)
            pca[atype][1] = pca[atype][1].to(device=device, non_blocking=non_block)
            pca[atype][2] = pca[atype][2].to(device=device, non_blocking=non_block)
    '''


    for item in batch:
        struct_weight.append(item['struct_weight'])
        tot_num.append(item['tot_num'])
        for atype in atom_types: 
            x[atype].append(item['x'][atype])
            tmp_dx = item['dx'][atype] #Make dx to sparse_tensor
            if use_stress:
                tmp_da = item['da'][atype]
            if tmp_dx.is_sparse:
                tmp_dx = tmp_dx.to_dense().reshape(item['dx_size'][atype])
            if scale_factor is not None:
                tmp_dx /= scale_factor[atype][1].view(1,-1,1,1)
                if use_stress:
                    tmp_da /= scale_factor[atype][1].view(1,-1,1,1)
            if pca is not None:
                if tmp_dx.size(0) != 0:
                    tmp_dx = torch.einsum('ijkl,jm->imkl', tmp_dx, pca[atype][0])
                    if use_stress:
                        tmp_da = torch.einsum('ijkl,jm->imkl', tmp_da, pca[atype][0])
                if pca_min_whiten_level is not None:
                    tmp_dx /= pca[atype][1].view(1,-1,1,1)
                    if use_stress:
                        tmp_da /= pca[atype][1].view(1,-1,1,1)
            dx[atype].append(tmp_dx) #sparse_tensor dx
            if use_stress:
                da[atype].append(tmp_da)
            n[atype].append(item['x'][atype].size(0))
        E.append(item['E'])
        F.append(item['F'])
        if use_stress:
            S.append(item['S'])

    for atype in atom_types:
        x[atype] = torch.cat(x[atype], axis=0)
        if scale_factor: #Scale part
            x[atype] -= scale_factor[atype][0].view(1,-1)
            x[atype] /= scale_factor[atype][1].view(1,-1)
        if pca: #PCA part
            if x[atype].size(0) != 0:
                #Important note 
                #tmp_dx mkatrix size should exceed number of symmetryfunction
                #If less than number of symmetry funtion -> n * n < # of SF matrix return
                x[atype] = torch.einsum('ij,jm->im', x[atype], pca[atype][0]) - pca[atype][2].reshape(1,-1)
            if pca_min_whiten_level is not None:
                x[atype] /= pca[atype][1].view(1,-1)
        n[atype] = torch.tensor(n[atype])
        #n[atype] = torch.tensor(n[atype], device=device)
        sparse_index[atype] = gen_sparse_index(n[atype])
        #sparse_index[atype] = gen_sparse_index(n[atype], device)
        
    struct_weight = torch.tensor(struct_weight) 
    tot_num = torch.tensor(tot_num)
    E = torch.tensor(E)
    #struct_weight = torch.tensor(struct_weight, device=device)  
    #tot_num = torch.tensor(tot_num, device=device)
    #E = torch.tensor(E, device=device)
 
    F = torch.cat(F, axis=0)
    if use_stress:
        S = torch.cat(S, axis=0)

    return {'x': x, 'dx': dx, 'da': da, 'n': n, 'E': E, 'F': F, 'S': S, 'sp_idx': sparse_index, 'struct_weight': struct_weight, 'tot_num': tot_num}

#Function to generate Iterator
def atomic_e_collate(batch, atom_types, scale_factor=None, pca=None, pca_min_whiten_level=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = dict()
    n = dict()
    sparse_index = dict()
    atomic_E = dict()
    atomic_num = dict()
    for atype in atom_types:
        x[atype] = list()
        n[atype] = list()
        sparse_index[atype] = list()
        atomic_E[atype] = list()
        atomic_num[atype] = list()
    E = list()
    struct_weight = list()
    tot_num = list()

    for item in batch:
        struct_weight.append(item['struct_weight'])
        tot_num.append(item['tot_num'])
        for atype in atom_types: 
            x[atype].append(item['x'][atype])
            n[atype].append(item['x'][atype].size(0))
            if atype in item['atomic_E'].keys():
                atomic_E[atype].append(item['atomic_E'][atype])
                atomic_num[atype].append(item['atomic_E'][atype].size(0))
            else:
                atomic_num[atype].append(0)
        E.append(item['E'])

    for atype in atom_types:
        x[atype] = torch.cat(x[atype], axis=0)
        if scale_factor: #Scale part
            x[atype] -= scale_factor[atype][0].view(1,-1)
            x[atype] /= scale_factor[atype][1].view(1,-1)
        if pca: #PCA part
            if x[atype].size(0) != 0:
                x[atype] = torch.einsum('ij,jm->im', x[atype], pca[atype][0]) - pca[atype][2].reshape(1,-1)
            if pca_min_whiten_level is not None:
                x[atype] /= pca[atype][1].view(1,-1)
        n[atype] = torch.tensor(n[atype])
        sparse_index[atype] = gen_sparse_index(n[atype])
        if atomic_E[atype]:
            atomic_E[atype] = torch.cat(atomic_E[atype])
        else:
            atomic_E[atype] = None
        atomic_num[atype] = torch.tensor(atomic_num[atype])
        
    E = torch.tensor(E)
    struct_weight = torch.tensor(struct_weight) 
    tot_num = torch.tensor(tot_num)

    return {'x': x,'n': n, 'E': E,'atomic_E' : atomic_E,'sp_idx': sparse_index, 'struct_weight': struct_weight, 'tot_num': tot_num, 'atomic_num' : atomic_num}

#Function to generate Iterator
def filename_collate(batch, atom_types, scale_factor=None, pca=None, pca_min_whiten_level=None, use_stress=False):
    tmp_dict = my_collate(batch, atom_types, scale_factor, pca, pca_min_whiten_level, use_stress)
    tmp_dict['filename'] = list()
    for item in batch:
        tmp_dict['filename'].append(item['filename'])
    return tmp_dict

def gen_sparse_index(nlist, device='cpu'):
    res = torch.zeros(2, sum(nlist), device=device)
    idx = 0
    for i,item in enumerate(nlist):
        for jtem in range(item):
            res[0, idx] = i
            res[1, idx] = idx
            idx += 1
    return res
 
#Load collate from train, valid dataset
def _make_dataloader(inputs, logfile, scale_factor, pca, train_dataset_list, valid_dataset_list, batch_size=1, my_collate=my_collate):

    if inputs['neural_network']['E_loss_type'] == 3:
        partial_collate = partial(
        atomic_e_collate, 
        atom_types=inputs['atom_types'], 
        scale_factor=scale_factor, 
        pca=pca, 
        pca_min_whiten_level=inputs['neural_network']['pca_min_whiten_level'])
    else:
        partial_collate = partial(
        my_collate, 
        #_my_collate_type1, 
        atom_types=inputs['atom_types'], 
        scale_factor=scale_factor, 
        pca=pca, 
        pca_min_whiten_level=inputs['neural_network']['pca_min_whiten_level'],
        use_stress=inputs['neural_network']['use_stress'])
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset_list, batch_size=batch_size, shuffle=inputs['descriptor']['shuffle'], collate_fn=partial_collate,
        #num_workers=inputs['neural_network']['workers'], pin_memory=False)
        num_workers=0, pin_memory=False)

    #Check test mode, valid dataset exist
    valid_loader = None
    if valid_dataset_list:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset_list, batch_size=batch_size, shuffle=inputs['descriptor']['shuffle'], collate_fn=partial_collate,
            #num_workers=inputs['neural_network']['workers'], pin_memory=False)
            num_workers=0, pin_memory=False)

    return train_loader, valid_loader


def delete_key_in_pt(filename, key):       
    filelist = FilelistDataset(filename)
    filelist.save_filename()  
    for f in range(filelist):
        pt_dict = f
        del pt_dict['filename']
        if key in pt_dict.keys():
            del pt_dict[key]
        torch.save(pt_dict, f['filename'])

