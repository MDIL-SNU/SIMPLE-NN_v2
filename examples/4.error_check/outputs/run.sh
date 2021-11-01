#!/bin/bash
PYTHON_DIR='python3 '

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
    ls ./data | sed 's/^/.\/data\//' > test_list
fi
#Copy saved model 
if [ ! -e ./checkpoint.tar ];then
    if [ -e ../3.train_model/checkpoint_bestmodel.pth.tar ];then
        cp ../3.train_model/checkpoint_bestmodel.pth.tar ./checkpoint.tar
    else
        echo saved model to load does not exist. Run 3.train_model procedure first to train model
    fi 
fi
#In this procedure test performance of trained model using generated data & checkpoint
#Checkpoint.tar is saved model that contains parameters of model
#You can also read model parameters from saved potential for lammps 
#by setting continue: 'weights' , potential_read
$PYTHON_DIR run.py

