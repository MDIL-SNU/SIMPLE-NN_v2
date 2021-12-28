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
- Uniform training for the biased sampling (https://doi.org/10.1021/acs.jpcc.8b08063)
- Replica ensemble for quantifying the uncertainty (https://doi.org/10.1021/acs.jpclett.0c01614)

## Requirement
- Python >= 3.6
- LAMMPS >= 29Oct2020

SIMPLE-NN can handle the output file of various quantum calculation codes such as `ABINIT`, `CASTEP`, `CP2K`, `Quantum espresso`, `Gaussian`, and `VASP` as dataset via atomic simulation environment ([ASE](https://wiki.fysik.dtu.dk/ase/index.html)) module.

Please visit [here](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) to check whether ASE module can read the output of your quantum calculation code or not. 
