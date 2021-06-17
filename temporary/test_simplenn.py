import sys
import os
sys.path.append('../')
import torch

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2 import run
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    run('input.yaml')

