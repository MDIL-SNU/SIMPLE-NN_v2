
<p align="left">
<img src="./docs/logo.png", width="300"/>
</p>
SIMPLE-NN (SNU Interatomic Machine-learning PotentiaL packagE – version Neural Network)

SIMPLE-NN is an open package that constructs Behler-Parrinello-type neural-network interatomic potentials from *ab initio* data. The package provides an interfacing module to LAMMPS for MD simulations. 

## Main features
- Training over total energies, forces, and stresses.
- Symmetry function vectors for atomic features
- PCA matrix transformation and whitening of training data for fast and accurate learning. 
- Uniform training for the biased sampling (https://doi.org/10.1021/acs.jpcc.8b08063)
- Replica ensemble for quantifying the uncertainty (https://doi.org/10.1021/acs.jpclett.0c01614)
- Compatible with results of most ab initio codes such as Quantum-Espresso and VASP via ASE module.

## Requirement
- Python: `3.6-3.9`
- LAMMPS: `29Oct2020` or later

Installation, manual, and full details: https://simple-nn-v2.readthedocs.io

If you use SIMPLE-NN, please cite the following paper: 
K. Lee, D. Yoo, W. Jeong, and S. Han, "SIMPLE-NN: An efficient package for training and executing neural-network interatomic potentials", Comp. Phys. Comm.  **242**, 95 (2019) https://doi.org/10.1016/j.cpc.2019.04.014.
