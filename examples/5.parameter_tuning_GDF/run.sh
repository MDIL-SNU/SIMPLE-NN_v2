#!/bin/bash
if [ ! -e run.py ];then
    cat << EOF > run.py
from simple_nn_v2 import run

run('input.yaml')
EOF
fi
#Copy generated data from before process 
#To test model need generate data files & test_list that contain directory of data
if [ ! -e ./data ];then
    if [ -e ../1.generate/data ];then
        cp -r ../1.generate/data  .
    else
        echo data to test do not exist. Run 1.generate procedure first to generate data
    fi
fi
python3 run.py

