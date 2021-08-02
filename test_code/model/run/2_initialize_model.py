import sys
sys.path.insert(0, '../../../')
from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.models import run

import torch
from torch.nn import Linear
from torchsummary import summary

logfile = open('LOG', 'w')
torch.set_default_dtype(torch.float64)
device = run._set_device()

logfile.write('Test initialize_model.py\n')
inputs = initialize_inputs('./input.yaml', logfile)
atom_types = inputs['atom_types']

model = run._initialize_model(inputs, logfile, device)
print('model keys')
print(model.keys)
print()

for elem in atom_types:
    print('[%s] Neural Network model'%elem)
    summary(model.nets[elem], input_size=(70,), dtypes=[torch.float64])
    print()
    for l in model.nets[elem].lin:
        print(l)
        if isinstance(l, Linear):
            print(l.weight)
            print(l.bias)
