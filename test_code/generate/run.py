import sys
sys.path.append('../../')

from simple_nn_v2 import Simple_nn
from simple_nn_v2.features.symmetry_function import Symmetry_function

# 1. Make data files
model = Simple_nn('input.yaml', descriptor=Symmetry_function())
descriptor = Symmetry_function()
descriptor.parent = model
descriptor.set_inputs()

print("Making data files...")
descriptor.generate()
print("Done!")

# 2. Check if values are same as SIMPLE-NN ver.1
import pickle
import torch

f=open('../test_data/SiO2/data1.pickle', 'rb')
d1=pickle.load(f)
f.close()

DIR = descriptor.inputs['save_directory']
if descriptor.inputs['save_to_pickle'] == True:
    f=open(DIR+'/data1.pickle', 'rb')
    d2=pickle.load(f)
    f.close()
else:
    d2=torch.load(DIR+'/data1.pt')

err_key = 0

def comp(d1, d2):
    if type(d1) == dict:
        for key in d1.keys():
            comp(d1[key], d2[key])
    elif type(d1) == int or type(d1) == float or type(d1) == str:
        if d1 != d2:
            err_key = 1
    else:
        comp_list(d1, d2)

def comp_list(l1, l2):
    try:  # if l1 is list, numpy array type
        len(l1)
        for i in range(len(l1)):
            comp_list(l1[i], l2[i])
    except:  # if l1 is single value
        if l1 != l2:
            err_key = 1

print("\nChecking data file values...")

comp(d1, d2)

if err_key == 0:
    print("COMPLET: generate data Ok!")
else:
    print("ERROR: Wrong value in data file")
