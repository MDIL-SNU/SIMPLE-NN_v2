#!/bin/bash
#SBATCH --nodelist=n008
#SBATCH --ntasks-per-node=40         # Cores per node
#SBATCH --partition=gpu          # Partition name (skylake)
##
#SBATCH --job-name="example"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)


## HPC ENVIRONMENT DON'T REMOVE THIS PART
. /etc/profile.d/TMI.sh
##

#mpiexec.hydra -np $SLURM_NTASKS /data/vasp4us/vasp6/odin_6.2.0/vasp_std
mpiexec.hydra -np $SLURM_NTASKS python run.py

#python ~/git/CGNNP/run.py
