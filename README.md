# SIMPLE-NN_v2
<p align="center">
<img src="./docs/logo.png", width="500"/>
</p>
SIMPLE-NN_v2(SNU Interatomic Machine-learning PotentiaL packagE – version Neural Network)

If you use SIMPLE-NN_v2, please cite this article: 

K. Lee, D. Yoo, W. Jeong, S. Han, SIMPLE-NN: An efficient package for training and executing neural-network interatomic potentials, *Computer Physics Communications* (2019), https://doi.org/10.1016/j.cpc.2019.04.014.

Here do we describe minimal instruction to run the example of SIMPLE-NN.
If you want more information such as tuning parameters, please visit our online manual(https://simple-nn-v2.readthedocs.io).

## Main features
SIMPLE-NN is Python code to construct the neural network interatomic potential that can be used in Large-scale Atomic/Molecular Massively Parall Simulator (LAMMPS) for atomic simulation.
- PCA matrix transformation and whitening
- Uniform training for the biased sampling
- Replica ensemble for quantifying the uncertainty

## Requirement
- Python >= 3.6
- LAMMPS >= 29Oct2020

SIMPLE-NN can handle the output file of various quantum calculation codes such as `ABINIT`, `CASTEP`, `CP2K`, `Quantum espresso`, `Gaussian`, and `VASP` as dataset via atomic simulation environment ([ASE](https://wiki.fysik.dtu.dk/ase/index.html)) module.

Please visit [here](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) to check whether ASE module can read the output of your quantum calculation code or not. 

## Usage
To use SIMPLE-NN_v2, 3 types of files (input.yaml, params_XX, structure_list) are required.

### input.yaml
Parameter list to control SIMPLE-NN code is listed in input.yaml. 
The simple example of input.yaml is described below:
```YAML
# input.yaml
generate_features: true
preprocess: true
train_model: true
params:
    Si: params_Si
    O: params_O

symmetry_function:
    type: symmetry_function
  
neural_network:
    optimizer:
        method: Adam
    nodes: 30-30
```

### params_XX
params_XX (XX means atom symbol included your target system) contains the coefficients of symmetry functions.
Please read this [paper](https://aip.scitation.org/doi/10.1063/1.3553717) for detailed functional form of symmetry function.
Each line contains coefficients for one symmetry function. The format is defined as following:

```bash
2 1 0 6.0 0.003214 0.0 0.0
2 1 0 6.0 0.035711 0.0 0.0
4 1 1 6.0 0.000357 1.0 -1.0
4 1 1 6.0 0.028569 1.0 -1.0
4 1 1 6.0 0.089277 1.0 -1.0
```

First column indicates the type of symmetry function. Currently 2, 4 and 5 are available. 

Second and third columns indicate the type index of neighboring atoms. The index follows the order defined in `params` in the `input.yaml`. Because the radial symmetry function (type 2) requires only one neighbor atom, the third column is set to zero.
```bash
params:
    Si: params_Si
    O: params_O
```

The remaining parameters represent cutoff distance, η, ζ, and λ in the symmetry function.

### structure_list
structure_list contains the path of reference data. The format is described below:

```
/path/to/output_file :
/path/to/output_file 100:2000:20
/path/to/{1..10}/output_file :
``` 
The first and second columns stand for the path of reference data and index, repsecitvely.
The detailed description of index format is explained [here](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) 

### Script for running SIMPLE-NN
After preparing input.yaml, params_XX and structure_list, you can run SIMPLE-NN using `run.py` written below:

```python
"""
Run the code below:
    python run.py
    or
    mpirun -np numproc run.py # if you install mpi4py

run.py:
"""

from simple_nn_v2 import run
run('input.yaml')
```

### Script for using neural network interatomic potential in LAMMPS
To execute atomic simulation using neural network potential in LAMMPS, `pair_style` and `pair_coeff` in the input script for LAMMPS are written like this:

The atomic symbols after the name of the potential file can be changed depending on your target system.
```bash
pair_style nn
pair_coeff * * potential_saved Si O
```

## Example
In examples folder, one can find MD trajectories of bulk SiO<sub>2</sub>, corresponding input files (input.yaml, params_Si, params_O and structure_list) and python script run.py. To use this example, one simply change the location in the 'structure_list' file and run 'python run.py' command.

