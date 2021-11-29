#!/bin/bash
PYTHON_DIR='python3 '
cat << EOF > run.py
import sys
sys.path.append('../../../')
from simple_nn_v2 import run
run('input_gdf.yaml')
EOF
#Copy generated data from before process 
#To test model need generate data files & test_list that contain directory of data
if [ ! -e ./data ];then
    if [ -e ../1.generate/data ];then
        cp -r ../1.generate/data  .
    else
        echo data to test do not exist. Run 1.generate procedure first to generate data
    fi
fi
#This command add gaussian density function value of Symmetry function to saved files
$PYTHON_DIR run.py
mv LOG LOG_gdf
cat << EOF > run.py
import sys
sys.path.append('../../../')
from simple_nn_v2 import run
run('input_train.yaml')
EOF
$PYTHON_DIR run.py
