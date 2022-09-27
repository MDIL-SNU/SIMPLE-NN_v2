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


