/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nn,PairNN)

#else

#ifndef LMP_PAIR_NN
#define LMP_PAIR_NN
#define _USE_MATH_DEFINES

#include "pair.h"
#include "pair_nn_simd_function.h"
#include "symmetry_functions_simd.h"

using namespace NN_SIMD_NS;

namespace LAMMPS_NS {

class PairNN : public Pair {

 private :

  struct Symc {
    double coefs[4]; // symmetry function coefficients
    double powtwo; //precalculated value of angular sym
    bool powint; //precalculated value of angular sym
    int inputVecNum;
  };

  struct VectorizedSymc {
    VectorizedSymc() {}
    ~VectorizedSymc() {
      _mm_free(mask);
      _mm_free(eta);
      _mm_free(Rs);
      _mm_free(lammda);
      _mm_free(powtwo);
      
      delete [] lammda_i;
      delete [] zeta;
    }
    int vector_len;
    int true_size;
    int tt_offset;
    double cutoffr;

    double* mask=nullptr;
    double* eta=nullptr; // -1 * coefs[1]
    double* Rs=nullptr; //coefs[2] -> shift for radial, zeta for angular
    double* lammda=nullptr; //coefs[3] 1 or -1
    double* powtwo=nullptr;

    //initialized but not used yet (could be required for above angular sym ALGO)
    int* lammda_i=nullptr; //coefs[3] 1 or -1
    int* zeta=nullptr;
  };

  struct Net {
    //determined by potential file
    int *nnode; // number of nodes in each layer
    int nlayer; // number of layers

    //Net parameters for mkl
    AlignedMultiArr weights;
    AlignedMultiArr weights_T;
    AlignedMultiArr bias;
    double *scale1; //vector size of nsym
    double *scale2; //""

    typedef void(*ActivationFunctions)(double*, double*, const int);
    ActivationFunctions *actifuncs; //Array of function pointer. maybe slower(can't be inlined)

    // not used for simd
    Symc **radialLists; //first index : type of element, second : parameter set
    Symc ***angularLists1; //fisrt, seconds : "", third : "" : G4
    Symc ***angularLists2; //fisrt, seconds : "", third : "" : G5

    VectorizedSymc *radialListsVec;
    VectorizedSymc **angularLists1Vec;
    VectorizedSymc **angularLists2Vec;

    int *radialIndexer;
    int **angular1Indexer;
    int **angular2Indexer;
  };

  int *map;  // mapping from atom types to elements
  Net *nets; // network parameters
  double cutmax; //not used
  double max_rc_ang;

  bool isG4=false;
  bool isG5=false;

  //init NN parameters from potential file called from coeff()
  void read_file(char *);
  //called from destructor & read_file
  void free_net(Net &);
  // calculate atomic energy & derivative of net called from compute()
  double evalNet(double *, double *, Net &);

  //void RadialVector(const int,const int,const int,const int,const double* , double* , double* );

  //G4

  void AngularVector1_simd(const int , const int , const int , const int , const int , const int ,const int, const double , const double , const double , const double* , const double* , const double* , const double* , double* , double* );

  void AngularVector2_simd(const int , const int , const int , const int , const int , const int ,const int, const double , const double , const double , const double* , const double* , const double* , const double* , double* , double* );

  void RadialVector_simd(const int , const int , const int , const int ,const int, const double , const double* , double* , double* );

  void ForceAssign_simd(double** , const double* , double* , const int , const int , const int* , const int );

 public:
  PairNN(class LAMMPS *);
  ~PairNN();
  void compute(int, int);

  // lammps pair global setting, pair_nn doesn't support this
  void settings(int, char **);
  //read Atom type string from input script & related coeff
  void coeff(int, char **);

  //unnecessary functions, I think
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
