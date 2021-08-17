import sys
sys.path.append('./')
from simple_nn_v2 import simple_nn
from simple_nn_v2 import init_inputs
import torch

logfile = open('LOG', 'w', 10)


rootdir='./test_input/'
print(f"read {rootdir}input_test.yaml")
inputs = init_inputs.initialize_inputs(rootdir+'input_test.yaml', logfile)


#torch.save(inputs,rootdir+"inputs_match")
print(f"load pregenerated result {rootdir}inputs_match")
inputs_match = torch.load(rootdir+"inputs_match")


print("comparison with result")
if inputs.keys() == inputs_match.keys():
    print("Both have same key")
else:
    raise Exception("Error occred : not consistant keys")

for key in inputs.keys():
    assert inputs[key] == inputs_match[key] , f"Error occured : result not match in {key}"
    print(f"{key} passed")
 

print("test initialize_inputs done")
