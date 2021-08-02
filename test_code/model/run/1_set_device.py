import sys
sys.path.insert(0, '../../../')
from simple_nn_v2.models import run

import torch

torch.set_default_dtype(torch.float64)
device = run._set_device()
print(device)
