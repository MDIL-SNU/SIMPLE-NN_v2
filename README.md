
<p align="left">
<img src="./docs/logo.png", width="200"/>
</p>
SIMPLE-NN (SNU Interatomic Machine-learning PotentiaL packagE – version Neural Network)

SIMPLE-NN is an open package that constructs Behler-Parrinello-type neural-network interatomic potentials from *ab initio* data. The package provides an interfacing module to LAMMPS for MD simulations. 

## Main features
- Training over total energies, forces, and stresses.
- Symmetry function vectors for atomic features.
- Supports LAMMPS for MD simulations.
- PCA matrix transformation and whitening of training data for fast and accurate learning. 
- Supports GPU via PyTorch.
- CPU parallelization of preprocessing training data via MPI for Python
- Uniform training to rectify sample bias ([W. Jeong et al. *J. Phys. Chem. C* **122**, 22790 (2018)](https://doi.org/10.1021/acs.jpcc.8b08063)).
- Replica ensemble for uncertainty estimation ([W. Jeong et al. *J. Phys. Chem. Lett.* **11**, 6090 (2020)](https://doi.org/10.1021/acs.jpclett.0c01614)).
- Compatible with results of most ab initio codes such as Quantum-Espresso and VASP via ASE module.
- Dropout technique for regularizing neural networks.
- Requires Python `3.6-3.9` and LAMMPS (`23Jun2022` or newer)

Installation, manual, and full details: https://simple-nn-v2.readthedocs.io

If you use SIMPLE-NN, please cite:  
K. Lee, D. Yoo, W. Jeong, and S. Han, "SIMPLE-NN: An efficient package for training and executing neural-network interatomic potentials", Comp. Phys. Comm.  **242**, 95 (2019) https://doi.org/10.1016/j.cpc.2019.04.014.


Below are for advanced users.
## Intel Accelerator
The filename extension simd refers to Intel-accelerated version of simulating molecular dynamics in SIMPLE-NN. By exploiting vector-matrix multiplication routines in SIMD and Intel MKL, overall speed up would be x3 to x3.5 times faster than the regular version.

### Requirements
+ Intel CPU that supports AVX
+ IntelCompiler 18.0.5 or newer
+ IntelMKL 2018.5.274 or newer
+ lammps 23 Jun 2022 - Update 1(stable)  

The accelerated version requires intel compiler, so we recommend that you compile lammps source with intel compiler & intel mpi(mpiicpc). You can also check detailed installation guide for lammps intel pacakage below.
https://docs.lammps.org/Speed_intel.html

! The code use AVX related functions from intel intrinsic, BLAS routine and vector mathematics from mkl. So older version of MKL, intel compiler support those feature would be ok. 

### Installation
cp {pair_nn_simd.cpp, pair_nn_simd.h, pair_nn_simd_function.h, symmetry_functions_simd.h} {lammps_source}/src/
cd {lammps_source}/src/
make intel_cpu_intelmpi

Please note that 'make intel_cpu_intelmpi' is an example of using Intel compiler for lammps. You may change some library path and compile flags if needed.
If you have original pair_nn.* files in your {lammps_source}/src, you must remove original files pair_nn.* from your {lammps_source}/src/ but {pair_nn_replica.cpp, pair_nn_replica.h, symmetry_functions.h}.

### Requirements for potential file
For acceleration, there are some assumptions for a potential file. A potential file should comply with following rules.
Symmetry function group refers to a group of vector components which have the same target atom specie(s). 
+ Vector components in the same symmetry function group should have same a cutoff radius.
+ Vector components in the same symmetry function group should be contiguous in potential file.
+ The value of zeta should be integer in angular symmetry functions.
(Not requirement) For the best speed-up, the number of symmetry functions should be a multiple of "4" since AVX instruction sets support 256bit(total 4 double value) SIMD,

### Current Issue
'clear' command inside lammps input script could cause problem.

### Further Acceleration
If you cpu supports AVX512 instruction set, you can use AVX512 by adding 

 -D \_\_AVX512F\_\_

to your Makefile's CCFLAGS. Besides its capacity, speed up respect to AVX is minor (< 1%). This is because the bottelneck of the accelerated code is not arithmetic but memory.


