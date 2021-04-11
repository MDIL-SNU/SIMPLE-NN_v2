from ...simple_nn_v2 import simple_nn
from ...simple_nn_v2.features import symmetry_function
from ...simple_nn_v2.models import neural_network

simple_nn.run('input.yaml', descriptor=symmetry_function, model=neural_network)
