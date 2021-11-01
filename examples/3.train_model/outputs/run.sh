#!/bin/bash
if [ ! -e run.py ];then
    cat << EOF > run.py
from simple_nn_v2 import run

run('input.yaml')
EOF
fi
#Copy generated data from before process
for file in data train_list valid_list scale_factor pca; do
    if [ ! -e $file ];then
        cp -r ../2.preprocess/$file  .
    fi
    if [ ! -e $file ];then
        echo $file is not exist. Run 2.preprocess procedure first
        exit 254
    fi
done
#This command split train valid set by valid_rate from input.yaml
#to train_list, valid_list
#Also create scaler from all dataset to ./scale_factor and Principle Component Analysis ./pca
#This option can turn on & off by input.yaml options
python3 run.py
