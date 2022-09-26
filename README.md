
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
IntelAccelerated version of pair_nn is for molecular dynamics part of SIMPLE-NN. By exploiting SIMD and MKL's vector-matrix multiplication routines, total speed up around 3~3.5 times can be expected

### Requirements
+ intel CPU supports AVX
+ IntelCompiler/18.0.5 or newer
+ IntelMKL/2018.5.274 or newer
+ lammps 29 Sep 2021 - Update 3 (stable) tested

Although only accelerated version of pair_nn require intel compiler and lammps itself provides various ways to compile source, We highly recommend you to compile whole lammps source with intel compiler & intel mpi(mpiicpc). You can check detailed installation guid for lammps intel pacakage below.

https://docs.lammps.org/Speed_intel.html

! The code use AVX related functions from intel intrinsic, BLAS routine and vector mathematics from mkl. So older version of MKL, intel compiler support those feature would be ok. 

### Installation
cp { files } {lammps_source}/src/
make intel_cpu_intelmpi

Note that intel_cpu_intelmpi is example for intel compile for lammps. You may have to change some library path and compile flags if needed.

### Potential Requirements
For acceleration, there were some assumption for potential file. Therefor potential file given for accelerated code shoud follow rules below.
+ Inside symmetry function vector, vector components which share same center atom specie, target atom specie, same symmetry function type form "symmetry function group"
+ Vector components in same symmetry function group should have same cutoff radius
+ Vector components in same symmetry function group should written contiguously in potential file
+ For angular symmetry function, zeta should be integer

### Current Issue
'clear' command inside lammps input script could cause problem. You can avoid use of that command by extracting loop written in lammps input script to shell script or something so to don't have to 'clear' running lammps program.

### Further Acceleration
If you cpu supports AVX512 instruction set, you can use AVX512 by adding 

 -D \_\_AVX512F\_\_

to your Makefile's CCFLAGS. Besides its capacity speed up respect to AVX is minor (< 1%) This is because the most time consuming part of calculation is not floating point arithmetic anymore.

SIMD parallelism works on "Symmetry Function Group". Since AVX instruction set supports 256bit(total 4 double value) SIMD, determining number of symmetry function inside one symmetry fuction group multiple of "4" would results in best speed up w.r.t original version.
