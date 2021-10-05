#!/bin/bash
if [ ! -e run.py ];then
    cat << EOF > run.py
from simple_nn_v2 import run

run('input.yaml')
EOF
fi
#Copy generated data from before process
if [ ! -e ./data ];then
    cp -r ../1.generate_data/data  .
fi
#Copy directory list of generated data 
if [ ! -e ./total_list ];then
    cp -r ../1.generate_data/total_list  .
fi
#Check file exist
for file in ./data total_list; do
    if [ ! -e $file ];then
        echo $file is not exist. Run 1.generate_data procedure first
    fi
done
#This command split train valid set by valid_rate from input.yaml
#to train_list, valid_list
#Also create scaler from all dataset to ./scale_factor and Principle Component Analysis ./pca
#This option can turn on & off by input.yaml options
python3 run.py

