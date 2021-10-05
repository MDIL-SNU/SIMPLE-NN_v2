#!/bin/bash
if [ ! -e run.py ];then
    cat << EOF > run.py
from simple_nn_v2 import run

run('input.yaml')
EOF
fi
#This command create symmetry function as torch.save files from structure_list
#and save their directory to total_list
python3 run.py

#reulst./data folder & files are omitted in outputs 
#due to large file size ( ~4Gb)
