#!/bin/bash
PYTHON_DIR='python3 '

if [ ! -e run.py ];then
    cat << EOF > run.py
from simple_nn_v2 import run

run('input.yaml')
EOF
fi
#Copy generated data from before process
if [ ! -e ./data ];then
    if [ -e ../1.generate_data/data ];then
        cp -r ../1.generate_data/data  .
    else
        echo data to preprocess do not exist. Run 1.generate_data procedure first
        exit 254
    fi 
fi
#Copy directory list of generated data 
if [ ! -e ./total_list ];then
    if [ -e ../1.generate_data/total_list ];then
        cp -r ../1.generate_data/total_list  .
    else
        ls ./data | sed 's/^/0:.\/data\//' > total_list
    fi 
fi
#This command split train valid set by valid_rate from input.yaml
#to train_list, valid_list
#Also create scaler from all dataset to ./scale_factor and Principle Component Analysis ./pca
#This option can turn on & off by input.yaml options
$PYTHON_DIR run.py

