import torch
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

# From pytorch .pt files (will be implemented soon)
class TorchStyleDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.data = torch.load(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FilelistDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filelist = list()
        with open(filename) as fil:
            for line in fil:
                #for item in list(braceexpand(line.strip())):
                temp_list = glob(line.strip())
                temp_list.sort()
                ## Weight check and copy
                for item in temp_list:
                    tmp_weight = int(TorchStyleDataset(item)['struct_weight'])
                    for _ in range(tmp_weight):
                        self.filelist.append(item)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        return torch.load(self.filelist[idx])

class StructlistDataset(FilelistDataset):
    def __init__(self):
        self.filelist = list()

#Struct data set to 
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

    # add scale, pca
    for item in batch:
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
        if scale_factor is not None: #Scale part
            x[atype] -= scale_factor[atype][0].view(1,-1)
            x[atype] /= scale_factor[atype][1].view(1,-1)
        if pca is not None: #PCA part
            if x[atype].size(0) != 0:
                #Important note 
                #tmp_dx mkatrix size should exceed number of symmetryfunction
                #If less than number of symmetry funtion -> n * n < # of SF matrix return
                x[atype] = torch.einsum('ij,jm->im', x[atype], pca[atype][0]) - pca[atype][2].reshape(1,-1) 
            if pca_min_whiten_level is not None:
                x[atype] /= pca[atype][1].view(1,-1)
        sparse_index[atype] = gen_sparse_index(n[atype])
        n[atype] = torch.tensor(n[atype])
    E = torch.tensor(E)
    F = torch.cat(F, axis=0)
    if use_stress:
        S = torch.cat(S, axis=0)

    return {'x': x, 'dx': dx, 'da': da, 'n': n, 'E': E, 'F': F, 'S': S, 'sp_idx': sparse_index}

def gen_sparse_index(nlist):
    res = torch.zeros(2, sum(nlist))
    idx = 0
    for i,item in enumerate(nlist):
        for jtem in range(item):
            res[0,idx] = i
            res[1,idx] = idx
            idx += 1
    return res
