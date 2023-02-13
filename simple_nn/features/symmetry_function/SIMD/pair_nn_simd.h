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

PairStyle(nn/intel,PairNNIntel)

#else

#ifndef LMP_PAIR_NN_INTEL
#define LMP_PAIR_NN_INTEL
#define _USE_MATH_DEFINES

#include "pair.h"
#include "pair_nn_simd_function.h"

using namespace NN_SIMD_NS;

namespace LAMMPS_NS {

  class PairNNIntel : public Pair {
    private :
      struct Symc {
        double coefs[4]; // symmetry function coefficients
        double powtwo; //precalculated value of angular sym
        bool powint; //precalculated value of angular sym
        int inputVecNum;
      };
      struct VectorizedSymc {
        VectorizedSymc() {}
        ~VectorizedSymc();
        int vector_len;
        int tt_offset;
        double cutoffr;

        double* mask=nullptr;
        double* eta=nullptr; // -1 * coefs[1]
        double* Rs=nullptr; //coefs[2] -> shift for radial, zeta for angular
        double* lammda=nullptr; //coefs[3] 1 or -1
        double* powtwo=nullptr;

        int uq_eta_size;
        double* uq_eta=nullptr;
        int* uq_eta_map=nullptr;

        int max_zeta;
        int* uq_zeta_lammda_map = nullptr;

        void init_radial_vecSymc(Symc* target, const int len);
        void init_angular_vecSymc(Symc* target, const int len);
      };

      struct Net {
        ~Net();
        int nelements;

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
      //double max_rc_ang;
      double max_rc_ang_sq;

      bool isG4=false;
      bool isG5=false;
      bool optimize_G4=true;
      bool optimize_G5=true;

      //init NN parameters from potential file called from coeff()
      void read_file(char *);
      void init_vectorizedSymc(Net& net, const int nelements);
      //void init_AngularVecSymc(VectorizedSymc& 
      //called from destructor & read_file
      //void free_net(Net &);
      // calculate atomic energy & derivative of net called from compute()
      double evalNet(double *, double *, Net &);

      //G4
      void AngularVector1_simd(const int , const int , const int , const int , const int , const int ,const int, const double , const double , const double , const double* , const double* , const double* , const double* , double* , double* );
      void AngularVector1_simd_AVX2(const int , const int , const int , const int , const int , const int ,const int, const double , const double , const double , const double* , const double* , const double* , const double* , double* , double* );

      void AngularVector2_simd(const int , const int , const int , const int , const int , const int ,const int, const double , const double , const double , const double* , const double* , const double* , const double* , double* , double* );
      void AngularVector2_simd_AVX2(const int , const int , const int , const int , const int , const int ,const int, const double , const double , const double , const double* , const double* , const double* , const double* , double* , double* );

      void RadialVector_simd(const int , const int , const int , const int ,const int, const double , const double* , double* , double* );

    public:
      PairNNIntel(class LAMMPS *);
      ~PairNNIntel();
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
