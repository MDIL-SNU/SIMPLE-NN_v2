/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
http://lammps.sandia.gov, Sandia National Laboratories
Steve Plimpton, sjplimp@sandia.gov

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: 
   ------------------------------------------------------------------------- */

#include "pair_nn_simd.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "fmt/core.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "math_extra.h"
#include "force.h"

#include <set>

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace LAMMPS_NS;

#define MAXLINE 50000
#define MINR 0.0001

/* ---------------------------------------------------------------------- */
// Constructor
PairNNIntel::PairNNIntel(LAMMPS *lmp) : Pair(lmp) {
  map = nullptr;
  nets = nullptr;
  init_simd();
}

/* ---------------------------------------------------------------------- */
// Destructor
PairNNIntel::~PairNNIntel()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }
  delete [] nets;
  fin_simd();
}

/* ---------------------------------------------------------------------- */

inline void PairNNIntel::RadialVector_simd(const int ielem, const int jelem, const int ii_idx, const int jj_idx,const int stride_nsym, const double rRij, const double* vecij, double* symvec, double* tmpf)
{
  VectorizedSymc* vsym = &nets[ielem].radialListsVec[jelem];
  const int vector_size = vsym->vector_len;

  if (rRij > vsym->cutoffr) return;

  double fc, dfdR;
  cutf2_noslot(rRij, vsym->cutoffr, fc, dfdR);

  simd_v fc_v = SIMD_set(fc);
  simd_v dfdR_v = SIMD_set(dfdR);

  const int tt_offset = vsym->tt_offset;

  simd_v rRij_v = SIMD_set(rRij);

  for (int s=0; s < vector_size; s+=SIMD_V_LEN) {
    simd_v mask = SIMD_load_aligned(&vsym->mask[s]);
    simd_v eta_v = SIMD_load_aligned(&vsym->eta[s]);
    simd_v Rs_v = SIMD_load_aligned(&vsym->Rs[s]);

    simd_v tmp_v = rRij_v - Rs_v;
    simd_v eta_tmp_v = tmp_v * eta_v;

    //mask here!
    simd_v expl_v = SIMD_exp(eta_tmp_v*tmp_v)*mask;
    simd_v deriv_v = expl_v*(fmadd(eta_tmp_v,fc_v*2,dfdR_v));

    simd_v G2_v = fc_v * expl_v;
    simd_v* symvec_v = (simd_v*)(&symvec[tt_offset+s]);
    symvec_v[0] = symvec_v[0] + G2_v;

    simd_v* tmpf_simd = (simd_v*)(&tmpf[tt_offset+s]);

    tmpf_simd[jj_idx + 0*stride_nsym] = tmpf_simd[jj_idx + 0*stride_nsym] + deriv_v * vecij[0];
    tmpf_simd[jj_idx + 1*stride_nsym] = tmpf_simd[jj_idx + 1*stride_nsym] + deriv_v * vecij[1];
    tmpf_simd[jj_idx + 2*stride_nsym] = tmpf_simd[jj_idx + 2*stride_nsym] + deriv_v * vecij[2];

    tmpf_simd[ii_idx + 0*stride_nsym] = tmpf_simd[ii_idx + 0*stride_nsym] - deriv_v * vecij[0];
    tmpf_simd[ii_idx + 1*stride_nsym] = tmpf_simd[ii_idx + 1*stride_nsym] - deriv_v * vecij[1];
    tmpf_simd[ii_idx + 2*stride_nsym] = tmpf_simd[ii_idx + 2*stride_nsym] - deriv_v * vecij[2];
  }
}

inline void PairNNIntel::AngularVector1_simd(const int ielem, const int jelem, const int kelem, const int ii_idx, const int jj_idx, const int kk_idx, const int stride_nsym, const double rRij, const double rRik, const double rRjk, const double* vecij, const double* vecik, const double* vecjk, const double* precal_ang, double* symvec, double* tmpf) {

  VectorizedSymc* vsym = &nets[ielem].angularLists1Vec[jelem][kelem];
  const double cutr = vsym->cutoffr;
  const int vector_size = vsym->vector_len;
  const int tt_offset = vsym->tt_offset;

  if (vector_size == 0 || rRij > cutr || rRjk > cutr || rRik > cutr) return;

  double precal_cutf[6];

  cutf2_noslot(rRij, cutr, precal_cutf[0], precal_cutf[1]);
  cutf2_noslot(rRik, cutr, precal_cutf[2], precal_cutf[3]);
  cutf2_noslot(rRjk, cutr, precal_cutf[4], precal_cutf[5]);

  simd_v fcij_rRij_2 = SIMD_set(rRij*precal_cutf[0]*2);
  simd_v fcik_rRik_2 = SIMD_set(rRik*precal_cutf[2]*2);
  simd_v fcjk_rRjk_2 = SIMD_set(rRjk*precal_cutf[4]*2);

  simd_v fcij = SIMD_set(precal_cutf[0]);
  simd_v fcik = SIMD_set(precal_cutf[2]);
  simd_v fcjk = SIMD_set(precal_cutf[4]);

  simd_v dfdRij = SIMD_set(precal_cutf[1]);
  simd_v dfdRik = SIMD_set(precal_cutf[3]);
  simd_v dfdRjk = SIMD_set(precal_cutf[5]);

  for (int s=0; s < vector_size; s+=SIMD_V_LEN) {
    simd_v mask = SIMD_load_aligned(&vsym->mask[s]);
    simd_v eta_v = SIMD_load_aligned(&vsym->eta[s]); //par[1]
    simd_v zeta_v = SIMD_load_aligned(&vsym->Rs[s]); //par[2]
    simd_v lammda_v = SIMD_load_aligned(&vsym->lammda[s]); //par[3]
    simd_v powtwo_v = SIMD_load_aligned(&vsym->powtwo[s]);
   
    simd_v cosv = lammda_v * precal_ang[1] + 1;
    simd_v powcos = SIMD_pow(cosv, zeta_v-1);

    simd_v expl = SIMD_exp(eta_v*precal_ang[0])*powtwo_v;
    simd_v expl_powcos = mask*expl*powcos;

    simd_v deriv_ij = expl_powcos*fcik*fcjk*(fmadd(fmadd(eta_v,fcij_rRij_2,dfdRij),cosv,lammda_v*zeta_v*fcij*precal_ang[2]));
    simd_v deriv_ik = expl_powcos*fcij*fcjk*(fmadd(fmadd(eta_v,fcik_rRik_2,dfdRik),cosv,lammda_v*zeta_v*fcik*precal_ang[3]));
    simd_v deriv_jk = expl_powcos*fcij*fcik*(fmadd(fmadd(eta_v,fcjk_rRjk_2,dfdRjk),cosv,lammda_v*zeta_v*fcjk*precal_ang[4]));

    simd_v G4_v = expl_powcos*cosv*fcij*fcik*fcjk;

    simd_v* symvec_v = (simd_v*)(&symvec[tt_offset+s]);
    symvec_v[0] = symvec_v[0] + G4_v;

    simd_v tmpd_ij_x = deriv_ij*vecij[0];
    simd_v tmpd_ij_y = deriv_ij*vecij[1];
    simd_v tmpd_ij_z = deriv_ij*vecij[2];
    simd_v tmpd_ik_x = deriv_ik*vecik[0];
    simd_v tmpd_ik_y = deriv_ik*vecik[1];
    simd_v tmpd_ik_z = deriv_ik*vecik[2];
    simd_v tmpd_jk_x = deriv_jk*vecjk[0];
    simd_v tmpd_jk_y = deriv_jk*vecjk[1];
    simd_v tmpd_jk_z = deriv_jk*vecjk[2];

    simd_v* tmpf_simd = (simd_v*)(&tmpf[tt_offset + s]);

    tmpf_simd[jj_idx + 0*stride_nsym] = tmpf_simd[jj_idx + 0*stride_nsym] + tmpd_ij_x - tmpd_jk_x;
    tmpf_simd[jj_idx + 1*stride_nsym] = tmpf_simd[jj_idx + 1*stride_nsym] + tmpd_ij_y - tmpd_jk_y;
    tmpf_simd[jj_idx + 2*stride_nsym] = tmpf_simd[jj_idx + 2*stride_nsym] + tmpd_ij_z - tmpd_jk_z;

    tmpf_simd[kk_idx + 0*stride_nsym] = tmpf_simd[kk_idx + 0*stride_nsym] + tmpd_ik_x + tmpd_jk_x;
    tmpf_simd[kk_idx + 1*stride_nsym] = tmpf_simd[kk_idx + 1*stride_nsym] + tmpd_ik_y + tmpd_jk_y;
    tmpf_simd[kk_idx + 2*stride_nsym] = tmpf_simd[kk_idx + 2*stride_nsym] + tmpd_ik_z + tmpd_jk_z;

    tmpf_simd[ii_idx + 0*stride_nsym] = tmpf_simd[ii_idx + 0*stride_nsym] - tmpd_ij_x - tmpd_ik_x;
    tmpf_simd[ii_idx + 1*stride_nsym] = tmpf_simd[ii_idx + 1*stride_nsym] - tmpd_ij_y - tmpd_ik_y;
    tmpf_simd[ii_idx + 2*stride_nsym] = tmpf_simd[ii_idx + 2*stride_nsym] - tmpd_ij_z - tmpd_ik_z;
  }
}

//Assume for every G4, uniquie eta <= 4
inline void PairNNIntel::AngularVector1_simd_AVX2(const int ielem, const int jelem, const int kelem, const int ii_idx, const int jj_idx, const int kk_idx, const int stride_nsym, const double rRij, const double rRik, const double rRjk, const double* vecij, const double* vecik, const double* vecjk, const double* precal_ang, double* symvec, double* tmpf) {

  VectorizedSymc* vsym = &nets[ielem].angularLists1Vec[jelem][kelem];
  const double cutr = vsym->cutoffr;
  const int vector_size = vsym->vector_len;
  const int tt_offset = vsym->tt_offset;

  if (vector_size == 0 || rRij > cutr || rRjk > cutr || rRik > cutr) return;

  double precal_cutf[6];

  cutf2_noslot(rRij, cutr, precal_cutf[0], precal_cutf[1]);
  cutf2_noslot(rRik, cutr, precal_cutf[2], precal_cutf[3]);
  cutf2_noslot(rRjk, cutr, precal_cutf[4], precal_cutf[5]);

  const double fcij = precal_cutf[0];
  const double fcik = precal_cutf[2];
  const double fcjk = precal_cutf[4];

  const double fcij_rRij_2 = rRij*fcij*2;
  const double fcik_rRik_2 = rRik*fcik*2;
  const double fcjk_rRjk_2 = rRjk*fcjk*2;

  const double dfdRij = precal_cutf[1];
  const double dfdRik = precal_cutf[3];
  const double dfdRjk = precal_cutf[5];

  //unique eta calc should be less than SIMD_V_LEN
  simd_v tmp = SIMD_exp(SIMD_load_aligned(&vsym->uq_eta[0])*precal_ang[0]);
  double* uq_expl = (double *)&tmp.v;

  //unique zeta-lammda pair calc
  const int max_zeta = vsym->max_zeta;
  const double cosv_base_p = 1+precal_ang[1];
  const double cosv_base_n = 1-precal_ang[1];
  double pow_cosv_p = cosv_base_p;
  double pow_cosv_n = cosv_base_n;
  double uq_powcos[16];
  uq_powcos[0] = 1;
  uq_powcos[1] = 1;
  uq_powcos[2] = cosv_base_p;
  uq_powcos[3] = cosv_base_n;

  //pow from 0 to max_zeta-1
  for (int s=2; s<max_zeta; s++) {
    pow_cosv_p *= cosv_base_p;
    pow_cosv_n *= cosv_base_n;
    uq_powcos[2*s] = pow_cosv_p;
    uq_powcos[2*s+1] = pow_cosv_n;
  }

  for (int s=0; s < vector_size; s+=SIMD_V_LEN) {
    simd_v mask = SIMD_load_aligned(&vsym->mask[s]);
    simd_v eta_v = SIMD_load_aligned(&vsym->eta[s]); //par[1]
    simd_v zeta_v = SIMD_load_aligned(&vsym->Rs[s]); //par[2]
    simd_v lammda_v = SIMD_load_aligned(&vsym->lammda[s]); //par[3]
    simd_v powtwo_v = SIMD_load_aligned(&vsym->powtwo[s]);
    
    simd_v cosv = lammda_v * precal_ang[1] + 1;
    simd_v powcos = SIMD_gather(uq_powcos, &vsym->uq_zeta_lammda_map[s]);
    simd_v expl = SIMD_gather(uq_expl, &vsym->uq_eta_map[s]);

    simd_v expl_powcos = mask*expl*powcos*powtwo_v;

    simd_v deriv_ij = expl_powcos*fcik*fcjk*(fmadd((eta_v*fcij_rRij_2+dfdRij),cosv,lammda_v*zeta_v*fcij*precal_ang[2]));
    simd_v deriv_ik = expl_powcos*fcij*fcjk*(fmadd((eta_v*fcik_rRik_2+dfdRik),cosv,lammda_v*zeta_v*fcik*precal_ang[3]));
    simd_v deriv_jk = expl_powcos*fcij*fcik*(fmadd((eta_v*fcjk_rRjk_2+dfdRjk),cosv,lammda_v*zeta_v*fcjk*precal_ang[4]));

    simd_v G4_v = expl_powcos*cosv*fcij*fcik*fcjk;

    simd_v* symvec_v = (simd_v*)(&symvec[tt_offset+s]);
    symvec_v[0] = symvec_v[0] + G4_v;

    simd_v tmpd_ij_x = deriv_ij*vecij[0];
    simd_v tmpd_ij_y = deriv_ij*vecij[1];
    simd_v tmpd_ij_z = deriv_ij*vecij[2];
    simd_v tmpd_ik_x = deriv_ik*vecik[0];
    simd_v tmpd_ik_y = deriv_ik*vecik[1];
    simd_v tmpd_ik_z = deriv_ik*vecik[2];
    simd_v tmpd_jk_x = deriv_jk*vecjk[0];
    simd_v tmpd_jk_y = deriv_jk*vecjk[1];
    simd_v tmpd_jk_z = deriv_jk*vecjk[2];

    simd_v* tmpf_simd = (simd_v*)(&tmpf[tt_offset + s]);

    tmpf_simd[jj_idx + 0*stride_nsym] = tmpf_simd[jj_idx + 0*stride_nsym] + tmpd_ij_x - tmpd_jk_x;
    tmpf_simd[jj_idx + 1*stride_nsym] = tmpf_simd[jj_idx + 1*stride_nsym] + tmpd_ij_y - tmpd_jk_y;
    tmpf_simd[jj_idx + 2*stride_nsym] = tmpf_simd[jj_idx + 2*stride_nsym] + tmpd_ij_z - tmpd_jk_z;

    tmpf_simd[kk_idx + 0*stride_nsym] = tmpf_simd[kk_idx + 0*stride_nsym] + tmpd_ik_x + tmpd_jk_x;
    tmpf_simd[kk_idx + 1*stride_nsym] = tmpf_simd[kk_idx + 1*stride_nsym] + tmpd_ik_y + tmpd_jk_y;
    tmpf_simd[kk_idx + 2*stride_nsym] = tmpf_simd[kk_idx + 2*stride_nsym] + tmpd_ik_z + tmpd_jk_z;

    tmpf_simd[ii_idx + 0*stride_nsym] = tmpf_simd[ii_idx + 0*stride_nsym] - tmpd_ij_x - tmpd_ik_x;
    tmpf_simd[ii_idx + 1*stride_nsym] = tmpf_simd[ii_idx + 1*stride_nsym] - tmpd_ij_y - tmpd_ik_y;
    tmpf_simd[ii_idx + 2*stride_nsym] = tmpf_simd[ii_idx + 2*stride_nsym] - tmpd_ij_z - tmpd_ik_z;
  }
}

inline void PairNNIntel::AngularVector2_simd(const int ielem, const int jelem, const int kelem, const int ii_idx, const int jj_idx, const int kk_idx, const int stride_nsym, const double rRij, const double rRik, const double rRjk, const double* vecij, const double* vecik, const double* vecjk, const double* precal_ang, double* symvec, double* tmpf) {
  VectorizedSymc* vsym = &nets[ielem].angularLists2Vec[jelem][kelem];
  const double cutr = vsym->cutoffr;
  const int vector_size = vsym->vector_len;
  const int tt_offset = vsym->tt_offset;

  if (vector_size == 0 || rRij > cutr || rRik > cutr) return;

  double precal_cutf[4];

  cutf2_noslot(rRij, cutr, precal_cutf[0], precal_cutf[1]);
  cutf2_noslot(rRik, cutr, precal_cutf[2], precal_cutf[3]);

  simd_v fcij_rRij_2 = SIMD_set(rRij*precal_cutf[0]*2);
  simd_v fcik_rRik_2 = SIMD_set(rRik*precal_cutf[2]*2);

  simd_v fcij = SIMD_set(precal_cutf[0]);
  simd_v fcik = SIMD_set(precal_cutf[2]);

  simd_v dfdRij = SIMD_set(precal_cutf[1]);
  simd_v dfdRik = SIMD_set(precal_cutf[3]);

  for (int s=0; s < vector_size; s+=SIMD_V_LEN) {
    simd_v mask = SIMD_load_aligned(&vsym->mask[s]);
    simd_v eta_v = SIMD_load_aligned(&vsym->eta[s]); //par[1]
    simd_v zeta_v = SIMD_load_aligned(&vsym->Rs[s]); //par[2]
    simd_v lammda_v = SIMD_load_aligned(&vsym->lammda[s]); //par[3]
    simd_v powtwo_v = SIMD_load_aligned(&vsym->powtwo[s]);

    simd_v cosv = lammda_v * precal_ang[1] + 1;
    simd_v powcos = SIMD_pow(cosv, zeta_v - 1);

    simd_v expl = SIMD_exp(eta_v*precal_ang[5])*powtwo_v;
    //mask here!
    simd_v expl_powcos = mask*expl*powcos;

    simd_v deriv_ij = expl_powcos*fcik*(fmadd(fmadd(eta_v,fcij_rRij_2,dfdRij),cosv,lammda_v*zeta_v*fcij*precal_ang[2]));
    simd_v deriv_ik = expl_powcos*fcij*(fmadd(fmadd(eta_v,fcik_rRik_2,dfdRik),cosv,lammda_v*zeta_v*fcik*precal_ang[3]));
    simd_v deriv_jk = expl_powcos*fcij*fcik*lammda_v*zeta_v*precal_ang[4];

    simd_v G5_v = expl_powcos*cosv*fcij*fcik;

    simd_v* symvec_v = (simd_v*)(&symvec[tt_offset+s]);
    symvec_v[0] = symvec_v[0] + G5_v;

    simd_v tmpd_ij_x = deriv_ij*vecij[0];
    simd_v tmpd_ij_y = deriv_ij*vecij[1];
    simd_v tmpd_ij_z = deriv_ij*vecij[2];
    simd_v tmpd_ik_x = deriv_ik*vecik[0];
    simd_v tmpd_ik_y = deriv_ik*vecik[1];
    simd_v tmpd_ik_z = deriv_ik*vecik[2];
    simd_v tmpd_jk_x = deriv_jk*vecjk[0];
    simd_v tmpd_jk_y = deriv_jk*vecjk[1];
    simd_v tmpd_jk_z = deriv_jk*vecjk[2];

    simd_v* tmpf_simd = (simd_v*)(&tmpf[tt_offset + s]);

    tmpf_simd[jj_idx + 0*stride_nsym] = tmpf_simd[jj_idx + 0*stride_nsym] + tmpd_ij_x - tmpd_jk_x;
    tmpf_simd[jj_idx + 1*stride_nsym] = tmpf_simd[jj_idx + 1*stride_nsym] + tmpd_ij_y - tmpd_jk_y;
    tmpf_simd[jj_idx + 2*stride_nsym] = tmpf_simd[jj_idx + 2*stride_nsym] + tmpd_ij_z - tmpd_jk_z;

    tmpf_simd[kk_idx + 0*stride_nsym] = tmpf_simd[kk_idx + 0*stride_nsym] + tmpd_ik_x + tmpd_jk_x;
    tmpf_simd[kk_idx + 1*stride_nsym] = tmpf_simd[kk_idx + 1*stride_nsym] + tmpd_ik_y + tmpd_jk_y;
    tmpf_simd[kk_idx + 2*stride_nsym] = tmpf_simd[kk_idx + 2*stride_nsym] + tmpd_ik_z + tmpd_jk_z;

    tmpf_simd[ii_idx + 0*stride_nsym] = tmpf_simd[ii_idx + 0*stride_nsym] - tmpd_ij_x - tmpd_ik_x;
    tmpf_simd[ii_idx + 1*stride_nsym] = tmpf_simd[ii_idx + 1*stride_nsym] - tmpd_ij_y - tmpd_ik_y;
    tmpf_simd[ii_idx + 2*stride_nsym] = tmpf_simd[ii_idx + 2*stride_nsym] - tmpd_ij_z - tmpd_ik_z;
  }
}

inline void PairNNIntel::AngularVector2_simd_AVX2(const int ielem, const int jelem, const int kelem, const int ii_idx, const int jj_idx, const int kk_idx, const int stride_nsym, const double rRij, const double rRik, const double rRjk, const double* vecij, const double* vecik, const double* vecjk, const double* precal_ang, double* symvec, double* tmpf) {
  VectorizedSymc* vsym = &nets[ielem].angularLists2Vec[jelem][kelem];
  const double cutr = vsym->cutoffr;
  const int vector_size = vsym->vector_len;
  const int tt_offset = vsym->tt_offset;

  if (vector_size == 0 || rRij > cutr || rRik > cutr) return;

  double precal_cutf[4];

  cutf2_noslot(rRij, cutr, precal_cutf[0], precal_cutf[1]);
  cutf2_noslot(rRik, cutr, precal_cutf[2], precal_cutf[3]);

  simd_v fcij_rRij_2 = SIMD_set(rRij*precal_cutf[0]*2);
  simd_v fcik_rRik_2 = SIMD_set(rRik*precal_cutf[2]*2);

  simd_v fcij = SIMD_set(precal_cutf[0]);
  simd_v fcik = SIMD_set(precal_cutf[2]);

  simd_v dfdRij = SIMD_set(precal_cutf[1]);
  simd_v dfdRik = SIMD_set(precal_cutf[3]);

  simd_v tmp = SIMD_exp(SIMD_load_aligned(&vsym->uq_eta[0])*precal_ang[5]);
  double* uq_expl = (double *)&tmp.v;

  //unique zeta-lammda pair calc
  const int max_zeta = vsym->max_zeta;
  const double cosv_base_p = 1+precal_ang[1];
  const double cosv_base_n = 1-precal_ang[1];
  double pow_cosv_p = cosv_base_p;
  double pow_cosv_n = cosv_base_n;
  double uq_powcos[16];
  uq_powcos[0] = 1;
  uq_powcos[1] = 1;
  uq_powcos[2] = cosv_base_p;
  uq_powcos[3] = cosv_base_n;

  //pow from 0 to max_zeta-1
  for (int s=2; s<max_zeta; s++) {
    pow_cosv_p *= cosv_base_p;
    pow_cosv_n *= cosv_base_n;
    uq_powcos[2*s] = pow_cosv_p;
    uq_powcos[2*s+1] = pow_cosv_n;
  }

  for (int s=0; s < vector_size; s+=SIMD_V_LEN) {
    simd_v mask = SIMD_load_aligned(&vsym->mask[s]);
    simd_v eta_v = SIMD_load_aligned(&vsym->eta[s]); //par[1]
    simd_v zeta_v = SIMD_load_aligned(&vsym->Rs[s]); //par[2]
    simd_v lammda_v = SIMD_load_aligned(&vsym->lammda[s]); //par[3]
    simd_v powtwo_v = SIMD_load_aligned(&vsym->powtwo[s]);

    simd_v cosv = lammda_v * precal_ang[1] + 1;
    simd_v powcos = SIMD_gather(uq_powcos, &vsym->uq_zeta_lammda_map[s]);

    simd_v expl = SIMD_gather(uq_expl, &vsym->uq_eta_map[s]);
    //mask here!
    simd_v expl_powcos = mask*expl*powcos*powtwo_v;

    simd_v deriv_ij = expl_powcos*fcik*(fmadd(fmadd(eta_v,fcij_rRij_2,dfdRij),cosv,lammda_v*zeta_v*fcij*precal_ang[2]));
    simd_v deriv_ik = expl_powcos*fcij*(fmadd(fmadd(eta_v,fcik_rRik_2,dfdRik),cosv,lammda_v*zeta_v*fcik*precal_ang[3]));
    simd_v deriv_jk = expl_powcos*fcij*fcik*lammda_v*zeta_v*precal_ang[4];

    simd_v G5_v = expl_powcos*cosv*fcij*fcik;

    simd_v* symvec_v = (simd_v*)(&symvec[tt_offset+s]);
    symvec_v[0] = symvec_v[0] + G5_v;

    simd_v tmpd_ij_x = deriv_ij*vecij[0];
    simd_v tmpd_ij_y = deriv_ij*vecij[1];
    simd_v tmpd_ij_z = deriv_ij*vecij[2];
    simd_v tmpd_ik_x = deriv_ik*vecik[0];
    simd_v tmpd_ik_y = deriv_ik*vecik[1];
    simd_v tmpd_ik_z = deriv_ik*vecik[2];
    simd_v tmpd_jk_x = deriv_jk*vecjk[0];
    simd_v tmpd_jk_y = deriv_jk*vecjk[1];
    simd_v tmpd_jk_z = deriv_jk*vecjk[2];

    simd_v* tmpf_simd = (simd_v*)(&tmpf[tt_offset + s]);

    tmpf_simd[jj_idx + 0*stride_nsym] = tmpf_simd[jj_idx + 0*stride_nsym] + tmpd_ij_x - tmpd_jk_x;
    tmpf_simd[jj_idx + 1*stride_nsym] = tmpf_simd[jj_idx + 1*stride_nsym] + tmpd_ij_y - tmpd_jk_y;
    tmpf_simd[jj_idx + 2*stride_nsym] = tmpf_simd[jj_idx + 2*stride_nsym] + tmpd_ij_z - tmpd_jk_z;

    tmpf_simd[kk_idx + 0*stride_nsym] = tmpf_simd[kk_idx + 0*stride_nsym] + tmpd_ik_x + tmpd_jk_x;
    tmpf_simd[kk_idx + 1*stride_nsym] = tmpf_simd[kk_idx + 1*stride_nsym] + tmpd_ik_y + tmpd_jk_y;
    tmpf_simd[kk_idx + 2*stride_nsym] = tmpf_simd[kk_idx + 2*stride_nsym] + tmpd_ik_z + tmpd_jk_z;

    tmpf_simd[ii_idx + 0*stride_nsym] = tmpf_simd[ii_idx + 0*stride_nsym] - tmpd_ij_x - tmpd_ik_x;
    tmpf_simd[ii_idx + 1*stride_nsym] = tmpf_simd[ii_idx + 1*stride_nsym] - tmpd_ij_y - tmpd_ik_y;
    tmpf_simd[ii_idx + 2*stride_nsym] = tmpf_simd[ii_idx + 2*stride_nsym] - tmpd_ij_z - tmpd_ik_z;
  }
}

double PairNNIntel::evalNet(double* inpv, double *outv, Net &net){
  int nl = net.nlayer;
  const int lastLayer = nl - 2;
  //for better readability!!!!
  int* nnode = net.nnode + 1;
  int ninpv = net.nnode[0];

  //indexing  :  0, 1, 2, 3 (nnode too)
  //nodes[3] is atomic energy (nnode[3] is 1)
  AlignedMultiArr nodes(nnode, nl-1);
  AlignedMultiArr bnodes(nnode, nl-1);
  AlignedMultiArr dnodes(nnode, nl-1);

  vdSub(ninpv, inpv, net.scale1, inpv);
  vdMul(ninpv, inpv, net.scale2, inpv);
  for (int i=0; i<=lastLayer; i++) {
    cblas_dcopy(nnode[i], net.bias[i], 1, nodes[i], 1);
  }

  //                                       nROW      nCOL   alpha  Matrix          LDA    Vec1  Vecinc beta Vec2      Vecinc
  cblas_dgemv(CblasRowMajor, CblasNoTrans, nnode[0], ninpv, 1.0,   net.weights[0], ninpv, inpv, 1,     1.0, nodes[0], 1); //inpv to nodes[0] (PCA preprocessing) (nnode[0] & ninpv is same)
  actifunc_linear_vectorized(nodes[0], dnodes[0], nnode[0]);

  //after preprocess, so l starts from 1
  //loop over l = 1, 2, 3 (nl = 5)
  for (int l=1; l<nl-1; l++) {
    //nodes[l-1] to nodes[l]
    cblas_dgemv(CblasRowMajor, CblasNoTrans, nnode[l], nnode[l-1], 1.0, net.weights[l], nnode[l-1], nodes[l-1], 1, 1.0, nodes[l], 1);
    if(l == lastLayer) {
      //last hidden to atomic E(always linear)
      actifunc_linear_vectorized(nodes[l], dnodes[l], nnode[l]);
    }
    else {
      //acti from hidden
      net.actifuncs[l](nodes[l], dnodes[l], nnode[l]);
    }
  }

  //last hidden to atomic E -> must be linear -> set to 1
  //std::fill(bnodes[lastLayer], bnodes[lastLayer] + nnode[lastLayer], 1);
  bnodes[lastLayer][0] = 1;

  //backprop from last to preprocessed inpv
  //counter part of forward
  for (int l=lastLayer; l>=1; l--) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, nnode[l-1], nnode[l], 1.0, net.weights_T[l], nnode[l], bnodes[l], 1, 0.0, bnodes[l-1], 1);
    vdMul(nnode[l-1], bnodes[l-1], dnodes[l-1], bnodes[l-1]);
  }

  cblas_dgemv(CblasRowMajor, CblasNoTrans, ninpv, nnode[0], 1.0, net.weights_T[0], nnode[0], bnodes[0], 1, 0.0, outv, 1);
  vdMul(ninpv, outv, net.scale2, outv);

  return nodes[lastLayer][0]; // atomic energy
}

void PairNNIntel::compute(int eflag, int vflag)
{
  //Array of Struct
  struct NeighInfo {
    int jelem;
    double R;
    double delij[3];
    NeighInfo() {}
  };

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;
  if(vflag_atom) {
    error->all(FLERR,"vflag_atom related feature is not supported (atomic stress)");
  }

  double **x = atom->x;
  double **f = atom->f;
  //alias
  int* type = atom->type;
  int inum = list->inum;
  int* ilist = list->ilist;
  int* numneigh = list->numneigh;
  int** firstneigh = list->firstneigh;

  int neigh_vec_size = numneigh[ilist[0]];
  std::vector<NeighInfo> ninfos(neigh_vec_size);
  std::vector<int> ang_ninfo_indexes(neigh_vec_size);
  std::vector<int> jj_used(neigh_vec_size);

  // loop over center atoms (MAIN loop)
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];

    const int itype = type[i];
    const int ielem = map[itype];
    const int* jlist = firstneigh[i];
    const int jnum = numneigh[i];

    if(jnum > neigh_vec_size){
      ninfos.resize(jnum);
      ang_ninfo_indexes.resize(jnum, 0);
      jj_used.resize(jnum);
      neigh_vec_size = jnum;
    }

    const int nsym = nets[ielem].nnode[0];

    int ninfo_indexer = 0;
    //NeighInfo * ninfos = new NeighInfo[jnum];

    int ang_ninfo_indexer = 0;
    int rad_to_ang_indexer = 0;
    //int * ang_ninfo_indexes= new int[jnum]{0}; // contain index of ninfos for angular symmetry function calc, init with 0
    
    //int * jj_used = new int[jnum];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      const int jtype = type[j];

      const double delij[3] = {x[j][0] - xtmp, x[j][1] - ytmp, x[j][2] - ztmp};
      const double Rij = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];

      if (Rij > cutsq[itype][jtype]) { continue; }
      else {
        NeighInfo& a = ninfos[ninfo_indexer];
        a.delij[0] = delij[0];
        a.delij[1] = delij[1];
        a.delij[2] = delij[2];
        a.R = Rij;
        a.jelem = map[jtype];
        jj_used[ninfo_indexer] = jj;

        if (Rij < max_rc_ang_sq) {
          ang_ninfo_indexes[ang_ninfo_indexer] = ninfo_indexer;
          ang_ninfo_indexer++;
        }
        ninfo_indexer++;
      }
    }
    const int stride_nsym = ((nsym-1)/SIMD_V_LEN) + 1;
    const int ii_idx = stride_nsym*3*ninfo_indexer;

    const int symvec_true_size = DATASIZE*(stride_nsym*SIMD_V_LEN+SIMD_V_LEN);
    const int tmpf_true_size = DATASIZE*(stride_nsym*SIMD_V_LEN*(ninfo_indexer+1)*3+SIMD_V_LEN); 

    double * symvec = static_cast<double*>(_mm_malloc(symvec_true_size, ALIGN_NUM));
    std::fill(symvec, symvec+symvec_true_size/DATASIZE, 0.0);

    // TODO: create inside evalNet and return pointer, delete after use. (for data locality)
    // is std::fill necessary?
    double * dsymvec = static_cast<double*>(_mm_malloc(symvec_true_size, ALIGN_NUM));
    std::fill(dsymvec, dsymvec+symvec_true_size/DATASIZE, 0.0);

    double * tmpf = static_cast<double*>(_mm_malloc(tmpf_true_size, ALIGN_NUM));
    std::fill(tmpf, tmpf+tmpf_true_size/DATASIZE, 0.0);

    // loop over i'th neighbor list
    for (int jn = 0; jn < ninfo_indexer; jn++) {
      NeighInfo& j_info = ninfos[jn];
      const double delij[3] = {j_info.delij[0], j_info.delij[1], j_info.delij[2]};
      const double Rij = j_info.R;
      const double jelem = j_info.jelem;

      const int jj_idx = stride_nsym*3*jn;

      const double rRij = sqrt(Rij);
      const double invRij = 1/rRij;
      const double vecij[3] = {delij[0]*invRij, delij[1]*invRij, delij[2]*invRij};

      RadialVector_simd(ielem, jelem, ii_idx, jj_idx, stride_nsym, rRij, vecij, symvec, tmpf);
      if(ang_ninfo_indexes[rad_to_ang_indexer] != jn) { continue; }

      rad_to_ang_indexer++;

      for (int akn = rad_to_ang_indexer; akn < ang_ninfo_indexer; akn++) {
        const int kn = ang_ninfo_indexes[akn];
        NeighInfo& k_info = ninfos[kn];
        const double delik[3] = {k_info.delij[0], k_info.delij[1], k_info.delij[2]};
        const double Rik = k_info.R;
        const double kelem = k_info.jelem;

        const int kk_idx = stride_nsym*3*kn;

        const double rRik = sqrt(Rik);

        const double invRik = 1/rRik;
        const double deljk[3] = {delik[0] - delij[0], delik[1] - delij[1], delik[2] - delij[2]};
        const double Rjk = deljk[0]*deljk[0] + deljk[1]*deljk[1] + deljk[2]*deljk[2];

        const double rRjk = sqrt(Rjk);
        const double invRjk = 1/rRjk;

        const double vecik[3] = {delik[0] * invRik, delik[1]*invRik, delik[2]*invRik};
        const double vecjk[3] = {deljk[0] * invRjk, deljk[1]*invRjk, deljk[2]*invRjk};

        //remember 1/Rik != invRik  (here rRik = 1/invRik = sqrt(Rik))
        const double precal_ang[6] = {Rij + Rik + Rjk, (Rij + Rik - Rjk)*(0.5*invRij*invRik), \
          0.5*(invRik+1/Rij*(Rjk*invRik - rRik)), 0.5*(invRij + 1/Rik*(Rjk*invRij - rRij)), \
            -rRjk*invRij*invRik, Rij + Rik};

        if (isG4) {
          if (optimize_G4) {
            AngularVector1_simd_AVX2(ielem, jelem, kelem, ii_idx, jj_idx, kk_idx, stride_nsym, rRij, rRik, rRjk, vecij, vecik, vecjk, precal_ang, symvec, tmpf);
          } else {
            AngularVector1_simd(ielem, jelem, kelem, ii_idx, jj_idx, kk_idx, stride_nsym, rRij, rRik, rRjk, vecij, vecik, vecjk, precal_ang, symvec, tmpf);
          }
        }
        if (isG5) {
          if (optimize_G5) {
            AngularVector2_simd_AVX2(ielem, jelem, kelem, ii_idx, jj_idx, kk_idx, stride_nsym, rRij, rRik, rRjk, vecij, vecik, vecjk, precal_ang, symvec, tmpf);
          } else {
            AngularVector2_simd(ielem, jelem, kelem, ii_idx, jj_idx, kk_idx, stride_nsym, rRij, rRik, rRjk, vecij, vecik, vecjk, precal_ang, symvec, tmpf);
          }
        }
      } //k atom loop
    } //j atom loop

    const double tmpE = evalNet(symvec, dsymvec, nets[ielem]); // atomic energy of i'th atom

    if (eflag_global) { eng_vdwl += tmpE; } // contribute to total energy
    if (eflag_atom) { eatom[i] += tmpE; }

    simd_v* dsym_v = (simd_v*)(dsymvec);
    simd_v* tmpf_v = (simd_v*)(tmpf);

    for (int p=0; p <= ninfo_indexer; p++) {
      const int n = (p!=ninfo_indexer)? jlist[jj_used[p]] : i;

      simd_v x_reduced = SIMD_set(0);
      simd_v y_reduced = SIMD_set(0);
      simd_v z_reduced = SIMD_set(0);
      //loop over nsym through simd-index
      for (int ts=0; ts<stride_nsym; ts++) {
        x_reduced = fmadd(tmpf_v[p*3*stride_nsym + 0*stride_nsym + ts] , dsym_v[ts], x_reduced);
        y_reduced = fmadd(tmpf_v[p*3*stride_nsym + 1*stride_nsym + ts] , dsym_v[ts], y_reduced);
        z_reduced = fmadd(tmpf_v[p*3*stride_nsym + 2*stride_nsym + ts] , dsym_v[ts], z_reduced);
      }
      for (int l=0; l<SIMD_V_LEN; l++) {
        f[n][0] -= x_reduced[l];
        f[n][1] -= y_reduced[l];
        f[n][2] -= z_reduced[l];
      }
    }

    //delete[] ninfos;
    //delete[] jj_used;
    //delete[] ang_ninfo_indexes;

    _mm_free(symvec);
    _mm_free(dsymvec);
    _mm_free(tmpf);
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */


void PairNNIntel::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

//more than setting coeff. it initialize PairNNIntel's member variables also
void PairNNIntel::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) {
    allocated = 1;
    int n = atom->ntypes;
    memory->create(setflag,n+1,n+1,"pair:setflag");
    for (int i = 1; i <= n; i++) {
      for (int j = i; j <= n; j++) {
        setflag[i][j] = 0;
      }
    }
    memory->create(cutsq,n+1,n+1,"pair:cutsq");
    map = new int[n+1];
  }

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }

  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  //nets = new Net[nelements+1]; // extra one is used for reading irrelevant elements.
  nets = new Net[nelements]; // extra one is used for reading irrelevant elements.

  // read potential file and initialize potential parameters
  read_file(arg[2]);

  n = atom->ntypes;
  for (i = 1; i <= n; i++) {
    for (j = i; j <= n; j++) {
      setflag[i][j] = 0;
    }
  }

  int max_nl = 0;
  int max_nnode = 0;
  for (int i=0; i<nelements; ++i) {
    nets[i].nelements = nelements;
    if(nets[i].nlayer > max_nl) {
      max_nl = nets[i].nlayer;
    }
    for (int j=0; j<max_nl; ++j) {
      if(nets[i].nnode[j] > max_nnode) {
        max_nnode = nets[i].nnode[j];
      }
    }
  }

  // clear setflag since coeff() called once with I,J = * *

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (i = 1; i <= n; i++) {
    for (j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

void PairNNIntel::read_file(char *fname) {
  int i,j,k;
  FILE *fp;
  if (comm->me == 0) {
    //read file
    fp = fopen(fname, "r");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open NN potential file %s",fname);
      error->one(FLERR,str);
    }
  }

  int n,nwords,nsym,nlayer,iscale,inode,ilayer,t_wb;
  int isym = 0;
  char line[MAXLINE], *ptr, *tstr;
  int eof = 0;
  int stats = 0;
  int nnet = 0; //element index for pair_nn::nets
  int max_sym_line = 6;
  int valid_count = 0;
  bool valid = false;
  const char* delm = " \t\n\r\f";
  int tempStype = 0;
  int tempAtype1 = 0;
  int tempAtype2 = 0;
  Symc sym_target = Symc();
  cutmax = 0;
  double max_rc_ang = 0.0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        if (stats != 1) error->one(FLERR,"insufficient potential");
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    // MPI_Bcast : what is this for?
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = utils::count_words(line);
    if (nwords == 0) continue;

    // get all potential parameters
    if (stats == 0) { // initialization
      //originally parsing first line of potential file. ex : ELEM_LIST Mg Ca O
      stats = 1;
    } else if (stats == 1) { // potential element setting
      tstr = strtok(line,delm); // POT
      char *t_elem = strtok(NULL,delm); // element string (ex : Mg)
      double t_cut = std::atof(strtok(NULL,delm)); // cutoff (float string -> float)
      if (t_cut > cutmax) cutmax = t_cut;

      for (i=0; i<nelements; i++) {
        if (strcmp(t_elem,elements[i]) == 0) {
          nnet = i;
          break;
        }
      }

      if (nnet == nelements) {
        if (valid) {
          //free_net(nets[nnet]); //error case should (already valid but not eof)
        }
        valid = true;
      }
      else {
        valid_count++;
      }

      // cutoff setting
      for (i=1; i<=atom->ntypes; i++) {
        if (map[i] == nnet) {
          for (j=1; j<=atom->ntypes; j++) {
            cutsq[i][j] = t_cut*t_cut;
          }
        }
      }
      stats = 2;

    } else if (stats == 2) { // symfunc number setting
      nets[nnet].radialIndexer = new int[nelements];
      nets[nnet].angular1Indexer = new int*[nelements];
      nets[nnet].angular2Indexer = new int*[nelements];
      for (i=0; i<nelements; i++) {
        nets[nnet].radialIndexer[i] = 0;
        nets[nnet].angular1Indexer[i] = new int[nelements];
        nets[nnet].angular2Indexer[i] = new int[nelements];
        for (j=0; j<nelements; j++) {
          nets[nnet].angular1Indexer[i][j] = 0;
          nets[nnet].angular2Indexer[i][j] = 0;
        }
      }

      tstr = strtok(line,delm);
      if (strncmp(tstr,"SYM",3) != 0)
        error->one(FLERR,"potential file error: missing info(# of symfunc)");

      nsym = std::atoi(strtok(NULL,delm));

      nets[nnet].scale1 = (double*)_mm_malloc(DATASIZE*nsym, ALIGN_NUM);
      nets[nnet].scale2 = (double*)_mm_malloc(DATASIZE*nsym, ALIGN_NUM);

      //nnet is number of different species in system
      //Todo : reduce its size fit, delete unnecessary list
      nets[nnet].radialLists = new Symc*[nelements];
      nets[nnet].angularLists1 = new Symc**[nelements];
      nets[nnet].angularLists2 = new Symc**[nelements];

      nets[nnet].radialListsVec = new VectorizedSymc[nelements];
      nets[nnet].angularLists1Vec = new VectorizedSymc*[nelements];
      nets[nnet].angularLists2Vec = new VectorizedSymc*[nelements];

      for(i=0; i<nelements; i++) {
        nets[nnet].angularLists1Vec[i] = new VectorizedSymc[nelements];
        nets[nnet].angularLists2Vec[i] = new VectorizedSymc[nelements];
      }

      for(i=0; i<nelements; i++) {
        nets[nnet].radialLists[i] = new Symc[nsym];
        nets[nnet].angularLists1[i] = new Symc*[nelements];
        nets[nnet].angularLists2[i] = new Symc*[nelements];
        for(j=0; j<nelements; j++) {
          nets[nnet].angularLists1[i][j] = new Symc[nsym];
          nets[nnet].angularLists2[i][j] = new Symc[nsym];
        }
      }

      stats = 3;
    } else if (stats == 3) { // read symfunc parameters, isym starts from 0
      sym_target = Symc(); //temporary
      sym_target.inputVecNum = isym;

      tempStype = atoi(strtok(line,delm)); // symmetry function type(2,4,5)

      for (i=0; i<4; i++) {
        sym_target.coefs[i] = atof(strtok(NULL,delm));
      }
      tstr = strtok(NULL,delm);
      for (i=0; i<nelements; i++) {
        if (strcmp(tstr, elements[i]) == 0) {
          tempAtype1 = i;
          break;
        }
      }
      if (tempStype >= 4) {
        // Find maximum cutoff distance among angular functions.
        max_rc_ang = std::max(max_rc_ang, sym_target.coefs[0]);

        tstr = strtok(NULL,delm);
        for (i=0; i<nelements; i++) {
          if (strcmp(tstr,elements[i]) == 0) {
            tempAtype2 = i;
            break;
          }
        }

        if (tempStype == 4) {
          int i1 = nets[nnet].angular1Indexer[tempAtype1][tempAtype2];
          int i2 = nets[nnet].angular1Indexer[tempAtype1][tempAtype2];
          nets[nnet].angularLists1[tempAtype1][tempAtype2][i1] = sym_target;
          nets[nnet].angularLists1[tempAtype2][tempAtype1][i2] = sym_target;
          nets[nnet].angular1Indexer[tempAtype1][tempAtype2]++;
          if (tempAtype1 != tempAtype2) {
            nets[nnet].angular1Indexer[tempAtype2][tempAtype1]++;
          }
        }
        else if (tempStype == 5) {
          int i1 = nets[nnet].angular2Indexer[tempAtype1][tempAtype2];
          int i2 = nets[nnet].angular2Indexer[tempAtype1][tempAtype2];
          nets[nnet].angularLists2[tempAtype1][tempAtype2][i1] = sym_target;
          nets[nnet].angularLists2[tempAtype2][tempAtype1][i2] = sym_target;
          nets[nnet].angular2Indexer[tempAtype1][tempAtype2]++;
          if (tempAtype1 != tempAtype2) {
            nets[nnet].angular2Indexer[tempAtype2][tempAtype1]++;
          }
        }
      } else {
        int index = nets[nnet].radialIndexer[tempAtype1];
        nets[nnet].radialLists[tempAtype1][index] = sym_target;
        nets[nnet].radialIndexer[tempAtype1]++;
      }

      //sanity check
      bool implemented = false;
      for (i=0; i < sizeof(IMPLEMENTED_TYPE) / sizeof(IMPLEMENTED_TYPE[0]); i++) {
        if (tempStype == IMPLEMENTED_TYPE[i]) {
          implemented = true;
          break;
        }
      }
      //if (!implemented) error->all(FLERR, "Not implemented symmetry function type!");

      if (tempStype == 4 || tempStype == 5) {
        if (sym_target.coefs[2] < 1.0) {
          error->all(FLERR, "Zeta in G4/G5 must be greater or equal to 1.0!");
        }
      }

      //check whether forward next chunk
      isym++;
      if (isym == nsym) {
        iscale = 0;
        isym = 0;
        stats = 4;
      }
    } else if (stats == 4) { // scale (consist of two line (scale1 scale2)
      tstr = strtok(line,delm);
      for (i=0; i<nsym; i++) {
        if(iscale==0) nets[nnet].scale1[i] = atof(strtok(NULL,delm));
        else if(iscale==1) nets[nnet].scale2[i] = 1/atof(strtok(NULL,delm));
      }
      iscale++;
      if (iscale == 2) {
        stats = 5;
      }
    } else if (stats == 5) { // network number setting & initializing
      /*
       * From here we parsing
       * Net 3 132 60 60 1
       */
      tstr = strtok(line,delm); //Net

      /*
       * For two hidden layer, there are actually 5 layer exists
       * (inpv) W_pca (inpv_pca) W (hidden1) W (hidden2) W (Atomic Energy)
       * In this example potential file's nlayer is 3
       */ 
      nlayer = atoi(strtok(NULL,delm)); //read number '3'
      nlayer += 1; //So +1 here (now 4)
      nets[nnet].nlayer = nlayer + 1; // True size of layer(5)

      nets[nnet].nnode = new int[nets[nnet].nlayer]; //total 5 layer, total 5 # of node
      nets[nnet].nnode[0] = nsym; // 0 & 1 layer is for pca preprocess, so # of nodes are same
      //and nnode[nlayer-1] is 1 ( atomic energy )
      ilayer = 1;

      while ((tstr = strtok(NULL,delm))) {
        //nnode[0] is initialized as 132(from inpv written top of potential file)
        //we start from index '1' here
        //than nnode[1] = 132(=ninpv), nnode[2] = 60, nnode[3] = 60, nnode[4] = 1
        nets[nnet].nnode[ilayer] = atoi(tstr);
        ilayer++;
      }

      //nets[nnet].nlayer - 1 (between two layers) is outer index of bias and weights
      int w_size[nlayer];
      int b_size[nlayer];
      //loop over 0 1 2 3
      for (i=0; i<nlayer; ++i) {
        w_size[i] = nets[nnet].nnode[i]*nets[nnet].nnode[i+1];
        b_size[i] = nets[nnet].nnode[i+1];
      }

      nets[nnet].weights.init(w_size, nlayer);
      nets[nnet].weights_T.init(w_size, nlayer);
      nets[nnet].bias.init(b_size, nlayer);
      //Initialize Array of function pointer here
      nets[nnet].actifuncs = new Net::ActivationFunctions[nlayer];

      stats = 6;
      ilayer = 0;
    } else if (stats == 6) { // layer setting
      tstr = strtok(line,delm); //LAYER
      tstr = strtok(NULL,delm); //integer (idk someting about layer index?)
      tstr = strtok(NULL,delm); //activation function string
      if (strncmp(tstr, "linear", 6) == 0) {
        nets[nnet].actifuncs[ilayer] = actifunc_linear_vectorized;
      }
      else if (strncmp(tstr, "sigmoid", 7) == 0) {
        nets[nnet].actifuncs[ilayer] = actifunc_sigmoid_vectorized;
      }
      else if (strncmp(tstr, "tanh", 4) == 0) {
        nets[nnet].actifuncs[ilayer] = actifunc_tanh_vectorized;
      }
      else if (strncmp(tstr, "relu", 4) == 0) {
        nets[nnet].actifuncs[ilayer] = actifunc_relu_vectorized;
      }
      else if (strncmp(tstr, "selu", 4) == 0) {
        nets[nnet].actifuncs[ilayer] = actifunc_selu_vectorized;
      }
      else if (strncmp(tstr, "swish", 5) == 0) {
        nets[nnet].actifuncs[ilayer] = actifunc_swish_vectorized;
      }
      else {
        nets[nnet].actifuncs[ilayer] = actifunc_linear_vectorized;
      }

      stats = 7;

      //somthing about indexing in next stats
      inode = 0;
      t_wb = 0;
    } else if (stats == 7) { // weights setting

      //for each node 1*bias + node*weights
      if (t_wb == 0) { // weights
        tstr = strtok(line,delm);
        //for each layer
        for (i=0; i<nets[nnet].nnode[ilayer]; i++) {
          nets[nnet].weights[ilayer][inode*nets[nnet].nnode[ilayer] + i] = atof(strtok(NULL,delm));
        }
        //similar to stats. go to nextline & read bias
        t_wb = 1;
      } else if (t_wb == 1) { // bias
        tstr = strtok(line,delm);
        nets[nnet].bias[ilayer][inode] = atof(strtok(NULL,delm));
        //back and forth
        t_wb = 0;
        //setting next node's weight & bias
        inode++;
      }

      if (inode == nets[nnet].nnode[ilayer+1]) {
        ilayer++;
        //look for next layer(read activation function type)
        stats = 6;
      }
      if (ilayer == nlayer) {
        //if (nnet == nelements) free_net(nets[nnet]); //error case(code can be removed)
        //look for next atom type
        stats = 1;
      }
    }
  } //while(1)
  if (valid_count == 0) error->one(FLERR,"potential file error: invalid elements");

  // pre-calculate some constants for symmetry functions.
  //loop through nets & all elemental pair of angular lists, pre calc powtwo
  for (int i=0; i<nelements; i++) {
    for (int j=0; j<nelements; j++) {
      for (int k=0; k<nelements; k++) {
        for (int t=0; t<nets[i].angular1Indexer[j][k]; t++) {
          nets[i].angularLists1[j][k][t].powtwo = pow(2, 1-nets[i].angularLists1[j][k][t].coefs[2]);
        }
        for (int t=0; t<nets[i].angular2Indexer[j][k]; t++) {
          nets[i].angularLists2[j][k][t].powtwo = pow(2, 1-nets[i].angularLists2[j][k][t].coefs[2]);
        }
      }
    }
  }

  //make transposed weights
  for(int i=0; i<nelements; i++) {
    for(int j=0; j<nets[i].nlayer-1; j++) {
      for(int k=0; k<nets[i].nnode[j+1]; k++) {
        for(int l=0; l<nets[i].nnode[j]; l++) {
          nets[i].weights_T[j][l*nets[i].nnode[j+1] + k] = nets[i].weights[j][k*nets[i].nnode[j] + l];
        }
      }
    }
  }

  max_rc_ang_sq = max_rc_ang*max_rc_ang;

  //copy assgined radial, angular lists info to vectorized version
  //loop over nets
  for (int i=0; i<nelements; i++) {
    init_vectorizedSymc(nets[i], nelements);
  }

  //cannot optimize if AVX2 is not availible
#ifndef __AVX2__
  //if (_may_i_use_cpu_feature(_FEATURE_AVX2) == false) {
  optimize_G4 = false;
  optimize_G5 = false;
  //}
#endif

  //print optimize status
  if (lmp->logfile) fprintf(lmp->logfile, "AVX2 for angular descriptor G4 calc : %s\n", optimize_G4 ? "on" : "off");
  if (lmp->logfile) fprintf(lmp->logfile, "AVX2 for angular descriptor G5 calc : %s\n", optimize_G5 ? "on" : "off");
#ifdef __AVX512F__
  if (lmp->logfile) fputs("AVX512 for descriptor calc : on\n", lmp->logfile);
#else
  if (lmp->logfile) fputs("AVX512 for descriptor calc : off\n", lmp->logfile);
#endif
}

void PairNNIntel::VectorizedSymc::init_radial_vecSymc(Symc* target, const int len) {
  const int pad_plus_size = len + SIMD_V_LEN;
  const int rad_true_size = (pad_plus_size)*DATASIZE;

  eta = (double*)_mm_malloc(rad_true_size, ALIGN_NUM);
  Rs = (double*)_mm_malloc(rad_true_size, ALIGN_NUM);
  mask = (double*)_mm_malloc(rad_true_size, ALIGN_NUM);

  vector_len = len;
  tt_offset = target[0].inputVecNum;

  if(len == 0) return;

  for (int s=0; s<len; s++) {
    Symc sym = target[s];
    cutoffr = sym.coefs[0];
    eta[s] = -sym.coefs[1];
    Rs[s] = sym.coefs[2];
    mask[s] = 1;
  }
  for (int s=len; s<pad_plus_size; s++) {
    eta[s] = 0;
    Rs[s] = 0;
    mask[s] = 0;
  }
}

void PairNNIntel::VectorizedSymc::init_angular_vecSymc(Symc* target, const int len) {
  const int pad_plus_size = len + SIMD_V_LEN;
  const int ang1_true_size = (pad_plus_size)*DATASIZE;
  vector_len = len;

  if(len == 0) return;

  eta = (double*)_mm_malloc(ang1_true_size, ALIGN_NUM);
  Rs = (double*)_mm_malloc(ang1_true_size, ALIGN_NUM);
  lammda = (double*)_mm_malloc(ang1_true_size, ALIGN_NUM);
  powtwo = (double*)_mm_malloc(ang1_true_size, ALIGN_NUM);
  mask = (double*)_mm_malloc(ang1_true_size, ALIGN_NUM);

  int * lammda_i = new int[len];
  int * zeta = new int[len];

  //uq_eta_map = new int[pad_plus_size];
  uq_eta_map = (int*)_mm_malloc(pad_plus_size*sizeof(int), ALIGN_NUM);
  uq_zeta_lammda_map = (int*)_mm_malloc(pad_plus_size*sizeof(int), ALIGN_NUM);
  
  tt_offset = target[0].inputVecNum;

  for (int s=0; s<len; s++) {
    Symc sym = target[s];
    cutoffr = sym.coefs[0];
    eta[s] = -sym.coefs[1];
    Rs[s] = sym.coefs[2];
    lammda[s] = sym.coefs[3];
    powtwo[s] = sym.powtwo;
    mask[s] = 1;

    lammda_i[s] = (int)sym.coefs[3];
    zeta[s] = (int)sym.coefs[2];
  }

  std::set<double> eta_set(eta, eta + len);
  uq_eta_size = eta_set.size();
  uq_eta = (double*)_mm_malloc((uq_eta_size+SIMD_V_LEN)*DATASIZE, ALIGN_NUM);
  std::copy(eta_set.begin(), eta_set.end(), uq_eta);

  for(int s=0; s<len; s++) for(int m=0; m<uq_eta_size; m++) {
    if(eta[s] == uq_eta[m]) {
      uq_eta_map[s] = m;
      break;
    }
  }
  //set padding elements of uq_eta to 0
  for (int s=uq_eta_size; s<uq_eta_size+SIMD_V_LEN; s++) {
    uq_eta[s] = 0;
  }

  max_zeta = *std::max_element(zeta, zeta+len);
  for (int s=0; s<len; s++) {
    if(lammda_i[s] == 1) {
      uq_zeta_lammda_map[s] = 2*(zeta[s] - 1);
    } else {
      uq_zeta_lammda_map[s] = 2*(zeta[s] - 1) + 1;
    }
  }

  //set padding elements value to 0
  for (int s=len; s<pad_plus_size; s++) {
    eta[s] = 0;
    Rs[s] = 0;
    lammda[s] = 0;
    powtwo[s] = 0;
    uq_eta_map[s] = 0;
    uq_zeta_lammda_map[s] = 0;
    mask[s] = 0;
  }
  delete[] lammda_i;
  delete[] zeta;
}

void PairNNIntel::init_vectorizedSymc(Net& net, const int nelements) {
  //loop over element
  for (int j=0; j<nelements; j++) {
    net.radialListsVec[j].init_radial_vecSymc(net.radialLists[j], net.radialIndexer[j]);

    for (int k=0; k<nelements; k++) {
      // unless # of vec is 0, have to initialize some trivial values
      net.angularLists1Vec[j][k].init_angular_vecSymc(net.angularLists1[j][k], net.angular1Indexer[j][k]);
      net.angularLists2Vec[j][k].init_angular_vecSymc(net.angularLists2[j][k], net.angular2Indexer[j][k]);

      if(net.angular1Indexer[j][k] > 0) {
          isG4 = true;
          if(net.angularLists1Vec[j][k].uq_eta_size > SIMD_V_LEN) optimize_G4 = false;
          if(net.angularLists1Vec[j][k].max_zeta > 8) optimize_G4 = false;
      }
      if(net.angular2Indexer[j][k] > 0) {
          isG5 = true;
          if(net.angularLists2Vec[j][k].uq_eta_size > SIMD_V_LEN) optimize_G5 = false;
          if(net.angularLists2Vec[j][k].max_zeta > 8) optimize_G5 = false;
      }
    } 
  } //jj
   
  if (isG4 == false) optimize_G4 = false;
  if (isG5 == false) optimize_G5 = false;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
   ------------------------------------------------------------------------- */

void PairNNIntel::init_style() {
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style nn requires newton pair on");    
  }

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   ------------------------------------------------------------------------- */

double PairNNIntel::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cutmax;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairNNIntel::write_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairNNIntel::read_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairNNIntel::write_restart_settings(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairNNIntel::read_restart_settings(FILE *fp) {}

/* ---------------------------------------------------------------------- */

double PairNNIntel::single(int i, int j, int itype, int jtype, double rsq,
    double factor_coul, double factor_lj,
    double &fforce)
{
  if (comm->me == 0) printf("single run\n");
  return factor_lj;
}

PairNNIntel::Net::~Net() {
  delete [] nnode;
  delete [] actifuncs;

  for (int i=0; i<nelements; i++) {
    for (int j=0; j<nelements; j++) {
      delete [] angularLists1[i][j];
      delete [] angularLists2[i][j];
    }
    delete [] angular1Indexer[i];
    delete [] angular2Indexer[i];
  
    delete [] angularLists1[i];
    delete [] angularLists2[i];
    delete [] radialLists[i];
  
    delete [] angularLists1Vec[i];
    delete [] angularLists2Vec[i];
  }
  delete [] angular1Indexer;
  delete [] angular2Indexer;
  delete [] radialIndexer;

  delete [] angularLists1;
  delete [] angularLists2;

  delete [] angularLists1Vec;
  delete [] angularLists2Vec;

  delete [] radialLists;
  delete [] radialListsVec;
  _mm_free(scale1);
  _mm_free(scale2);
}

PairNNIntel::VectorizedSymc::~VectorizedSymc() {
  if(mask != nullptr) {
    _mm_free(mask);
  }
  if(eta != nullptr) {
    _mm_free(eta);
  }
  if(Rs != nullptr) {
    _mm_free(Rs);
  }
  if(lammda != nullptr) {
    _mm_free(lammda);
  }
  if(powtwo != nullptr) {
    _mm_free(powtwo);
  }
  _mm_free(uq_eta);
  _mm_free(uq_eta_map);
  _mm_free(uq_zeta_lammda_map);
}

/* ---------------------------------------------------------------------- */


