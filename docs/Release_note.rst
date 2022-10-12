.. _Release_note:

Release note
============

v2.1.0 (29 Sep 2022)
-------------------
Breaking changes:

- Accelerated version of simulating molecular dynamics using SIMD and MKL.
    - Main code developer (Yutack Park).
    - Total 3x ~ 3.5x speed-up : :code:`pair_nn_simd.cpp`, :code:`pair_nn_simd.h` and :code:`pair_nn_simd_function.h`
    
v2.0.0 (3 Dec 2021)
-------------------
Breaking changes:

- Refactoring SIMPLE-NN from Tensorflow to PyTorch!
    - Main code developer (Seungwoo Hwang).
    - Main code developer (Sangmin Oh).
    - Project advisor and code developer (Jisu Jung).
    - Project organizer and original code developer (Kyuhyun Lee).

v1.1.1 (23 Sep 2021)
---------------------
General changes:

- Independent tags of :code:`generate` and :code:`preprocess` in :code:`input.yaml` for consistency (Jisu Jung).
- Extended buffer in LAMMPS potential (:code:`pair_nn.cpp`) for multinary (> 4) system (Jisu Jung).

Bug fixes:

- Fixed the inconsistency between the direct and cartesian positions from :code:`ASE` (Jisu Jung).
- Fixed the memory leak in LAMMPS potential (:code:`pair_nn.cpp`) (Jisu Jung).

v1.1.0 (13 Oct 2020)
---------------------
Development:

- Replica ensemble for quantifying the uncertainty (Wonseok Jeong and Jisu Jung).

Bug fixes:

- Fixed type mismatch in LAMMPS potential (:code:`pair_nn.cpp`) (Jisu Jung).

v1.0.0 (21 Feb 2020)
---------------------
Development:

- Stress training (Jisu Jung).

General changes:

- Optimized LAMMPS potential (:code:`pair_nn.cpp`) (Dongsun Yoo).
- Changed the unit in LOG from epoch to iteration (Dongsun Yoo).
- PCA whitening (Dongsun Yoo).

v0.8.0 (6 Apr 2019)
-------------------
Development:

- Gaussian density function (GDF) calculation (Kyuhyun Lee, Dongsun Yoo, Wonseok Jeong).

General changes:

- Added information to LOG (Kyuhyun Lee, Dongsun Yoo).

Bug fixes:

- Fixed MPI issues (Kyuhyun Lee, Dongsun Yoo).

v0.6.0 (20 Nov 2018)
--------------------
General changes:

- Added brace expansion in :code:`str_list` (Kyuhyun Lee).
- Added early stopping feature (Kyuhyun Lee).

v0.5.0 (11 Oct 2018)
--------------------
General changes:

- :code:`stddev` for weight initialization (Dongsun Yoo).

v0.4.6 (18 Sep 2018)
--------------------
General changes:

- Warning on undefined tag in :code:`input.yaml` (Dongsun Yoo).

Bug fixes:

- Fixed regularization. (Dongsun Yoo).

v0.4.5 (4 Sep 2018)
-------------------
General changes:

- User-defined optimizer (Kyuhyun Lee).

v0.4.3 (24 Aug 2018)
--------------------
General changes:

- Changed saving mechanism (Kyuhyun Lee).
