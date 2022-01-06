#!/bin/bash

# Running SIMPLE-NN
python run.py
if [ "$?" == 0 ]; then
    echo SIMPLE-NN test is passed.
else
    echo SIMPLE-NN test is failed.
fi

# Running LAMMPS with neural network
$1 -in nnp.in -screen none
if [ "$?" == 0 ]; then
    echo LAMMPS with neural network test is passed.
else
    echo LAMMPS with neural network test is failed.
fi

# Running LAMMPS with replica ensemble
$1 -in replica.in -screen none
if [ "$?" == 0 ]; then
    echo LAMMPS with replica ensemble test is passed.
else
    echo LAMMPS with replica ensemble test is failed.
fi
