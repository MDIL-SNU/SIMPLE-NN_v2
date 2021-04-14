import sys
sys.path.append('../')

from simple_nn_v2 import simple_nn

simple_nn.run('input.yaml', descriptor='symmetry_function', model='neural_network')
