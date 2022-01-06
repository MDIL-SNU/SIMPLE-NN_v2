#!/bin/bash

# Running SIMPLE-NN
python run.py
if [ "$?" == 0 ];then
    echo SIMPLE-NN test passed
else
    echo SIMPLE-NN test failed
fi

# Running LAMMPS
/path/to/lammps/src/lmp_mpi < nnp.in > lammps.out
if [ "$?" == 0 ];then
    echo LAMMPS test passed
else
    echo LAMMPS test failed
fi

# Running LAMMPS with replica ensemble
/path/to/lammps/src/lmp_mpi < uncertainty.in > lammps.out
if [ "$?" == 0 ];then
    echo LAMMPS with replica ensemble test passed
else
    echo LAMMPS with replica ensemble test failed
fi
