
<p align="left">
<img src="./docs/logo.png", width="300"/>
</p>
SIMPLE-NN (SNU Interatomic Machine-learning PotentiaL packagE – version Neural Network)

SIMPLE-NN is an open package that constructs Behler-Parrinello-type neural-network interatomic potentials from *ab initio* data. The package provides an interfacing module to LAMMPS for MD simulations. 

## Main features
- PCA matrix transformation and whitening
- Uniform training for the biased sampling (https://doi.org/10.1021/acs.jpcc.8b08063)
- Replica ensemble for quantifying the uncertainty (https://doi.org/10.1021/acs.jpclett.0c01614)

## Requirement
- Python: `3.6-3.9`
- LAMMPS: `29Oct2020` or later

SIMPLE-NN can handle the output file of various quantum calculation codes such as `ABINIT`, `CASTEP`, `CP2K`, `Quantum espresso`, `Gaussian`, and `VASP` as dataset via atomic simulation environment ([ASE](https://wiki.fysik.dtu.dk/ase/index.html)) module.

Please visit [here](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) to check whether ASE module can read the output of your quantum calculation code or not. 

Please visit our online manual for detailed descriptions of code (https://simple-nn-v2.readthedocs.io).

## Citation
If you use SIMPLE-NN, please cite this article: 

K. Lee, D. Yoo, W. Jeong, S. Han, SIMPLE-NN: An efficient package for training and executing neural-network interatomic potentials, *Computer Physics Communications* (2019), https://doi.org/10.1016/j.cpc.2019.04.014.
