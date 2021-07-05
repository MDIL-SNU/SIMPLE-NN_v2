import torch
import numpy as np
import shutil

class FCNDict(torch.nn.Module):
    def __init__(self, nets):
        super(FCNDict, self).__init__()
        self.nets = torch.nn.ModuleDict(nets)
        self.keys = self.nets.keys()

    def forward(self, x):
        assert [item for item in self.nets.keys()].sort() == [item for item in x.keys()].sort()
        res = {}
        for key in x:
            res[key] = self.nets[key](x[key])

        return res

    def write_lammps_potential(self, filename, inputs, scale_factor=None, pca=None):
        
        # TODO: get the parameter info from initial batch generting processs
        atom_type_str = ' '.join(inputs['atom_types'])

        FIL = open(filename, 'w')
        FIL.write('ELEM_LIST ' + atom_type_str + '\n\n')

        for item in inputs['atom_types']:
            params = list()
            with open(inputs['descriptor']['params'][item]) as fil:
                for line in fil:
                    tmp = line.split()
                    params += [list(map(float, tmp))]
            params = np.array(params)

            FIL.write('POT {} {}\n'.format(item, np.max(params[:,3])))
            FIL.write('SYM {}\n'.format(len(params)))

            for ctem in params:
                tmp_types = inputs['atom_types'][int(ctem[1]) - 1]
                if int(ctem[0]) > 3:
                    tmp_types += ' {}'.format(inputs['atom_types'][int(ctem[2]) - 1])
                if len(ctem) != 7:
                    raise ValueError("params file must have lines with 7 columns.")

                FIL.write('{} {} {} {} {} {}\n'.\
                    format(int(ctem[0]), ctem[3], ctem[4], ctem[5], ctem[6], tmp_types))

            if scale_factor is None:
                with open(inputs['descriptor']['params'][item],'r') as f:
                    tmp = f.readlines()
                input_dim= len(tmp) #open params read input number of symmetry functions
                FIL.write('scale1 {}\n'.format(' '.join(np.zeros(input_dim).astype(np.str))))
                FIL.write('scale2 {}\n'.format(' '.join(np.ones(input_dim).astype(np.str))))
            else:
                FIL.write('scale1 {}\n'.format(' '.join(scale_factor[item][0].cpu().numpy().astype(np.str))))
                FIL.write('scale2 {}\n'.format(' '.join(scale_factor[item][1].cpu().numpy().astype(np.str))))

            #weights = sess.run(self.models[item].weights)
            #nlayers = len(self.nodes[item])
            # An extra linear layer is used for PCA transformation.
            nodes = list()
            weights = list()
            biases = list()
            for n, i in self.nets[item].lin.named_modules():
                if 'lin' in n:
                    nodes.append(i.weight.size(0))
                    weights.append(i.weight.detach().cpu().numpy())
                    biases.append(i.bias.detach().cpu().numpy())
            #nodes.append(1)
            nlayers = len(nodes)
            if pca is not None:
                nodes = [pca[item][0].cpu().numpy().shape[1]] + nodes
                joffset = 1
            else:
                joffset = 0
            FIL.write('NET {} {}\n'.format(len(nodes)-1, ' '.join(map(str, nodes))))

            # PCA transformation layer.
            if pca is not None:
                FIL.write('LAYER 0 linear PCA\n')
                pca_mat = np.copy(pca[item][0].cpu().numpy())
                pca_mean = np.copy(pca[item][2].cpu().numpy())
                if inputs['neural_network']['pca_min_whiten_level'] is not None:
                    pca_mat /= pca[item][1].cpu().numpy().reshape([1, -1])
                    pca_mean /= pca[item][1].cpu().numpy()

                for k in range(pca[item][0].cpu().numpy().shape[1]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(pca_mat[:,k].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, -pca_mean[k]))

            for j in range(nlayers):
                # FIXME: add activation function type if new activation is added
                if j == nlayers-1:
                    acti = 'linear'
                else:
                    acti = inputs['neural_network']['acti_func']

                FIL.write('LAYER {} {}\n'.format(j+joffset, acti))

                for k in range(nodes[j + joffset]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(weights[j][k,:].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, biases[j][k]))

            FIL.write('\n')

        FIL.close()
        
class FCN(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, acti_func='sigmoid', dropout=None):
        super(FCN, self).__init__()

        self.lin = torch.nn.Sequential()

        dim_in = dim_input
        for i, hn in enumerate(dim_hidden):
            if dropout:
                self.lin.add_module(f'drop_{i}', torch.nn.Dropout(p=dropout))
            self.lin.add_module(f'lin_{i}', torch.nn.Linear(dim_in, hn))
           #if batch_norm:
            #    seq.add_module(torch.nn.BatchNorm1d(hn))
            dim_in = hn
            if acti_func == 'sigmoid':
                self.lin.add_module(f'sigmoid_{i}', torch.nn.Sigmoid())
            elif acti_func == 'tanh':
                self.lin.add_module(f'tanh_{i}', torch.nn.Tanh())
            elif acti_func == 'relu':
                self.lin.add_module(f'relu_{i}', torch.nn.ReLU())
            elif acti_func == 'selu':
                self.lin.add_module(f'tanh_{i}', torch.nn.SELU())
            elif acti_func == 'swish':
                self.lin.add_module(f'swish_{i}', swish())
            
        self.lin.add_module(f'lin_{i+1}', torch.nn.Linear(hn, 1))

    def forward(self, x):
        return self.lin(x)


class swish(torch.nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)


def read_lammps_potential(filename):
    def _read_until(fil, stop_tag):
        while True:
            line = fil.readline()
            if stop_tag in line:
                break

        return line

    shutil.copy2(filename, 'potential_read')

    weights = dict()
    with open(filename) as fil:
        atom_types = fil.readline().replace('\n', '').split()[1:]
        for item in atom_types:
            weights[item] = dict()            

            dims = list()
            dims.append(int(_read_until(fil, 'SYM').split()[1]))

            hidden_to_out = map(lambda x: int(x), _read_until(fil, 'NET').split()[2:])
            dims += hidden_to_out

            num_weights = len(dims) - 1
    
            tmp_idx = 0
            for j in range(num_weights):
                weights[item][f'lin_{tmp_idx}'] = dict()
                tmp_weights = np.zeros([dims[j], dims[j+1]])
                tmp_bias = np.zeros([dims[j+1]])

                # Since PCA will be dealt separately, skip PCA layer.
                skip = True if fil.readline().split()[-1] == 'PCA' else False
                for k in range(dims[j+1]):
                    tmp_weights[:,k] = list(map(lambda x: float(x), fil.readline().split()[1:]))
                    tmp_bias[k] = float(fil.readline().split()[1])

                weights[item][f'lin_{tmp_idx}']['weight'] = np.copy(tmp_weights)
                weights[item][f'lin_{tmp_idx}']['bias']  = np.copy(tmp_bias)
                if skip:
                    continue
                else:
                    tmp_idx += 1
    return weights
