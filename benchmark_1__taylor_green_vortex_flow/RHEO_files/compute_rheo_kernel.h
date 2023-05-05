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

#ifdef COMPUTE_CLASS

ComputeStyle(rheo/kernel,ComputeRHEOKernel)

#else

#ifndef LMP_COMPUTE_RHEO_KERNEL_H
#define LMP_COMPUTE_RHEO_KERNEL_H

#include "compute.h"
#include <unordered_map>
#include <unordered_set>

namespace LAMMPS_NS {

class ComputeRHEOKernel : public Compute {
 public:
  ComputeRHEOKernel(class LAMMPS *, int, char **);
  ~ComputeRHEOKernel();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double memory_usage();
  double calc_w(int,int,double,double,double,double);
  double calc_dw(int,int,double,double,double,double);
  int calc_d2w(int,int,double,double,double,double);
  double calc_w_quintic(int,int,double,double,double,double);
  double calc_dw_quintic(int,int,double,double,double,double,double *,double *);
  double dWij[3], dWji[3], d2Wij[3], d2Wji[3], Wij, Wji;
  void create_memory();
  void refresh_memory();
  int memory_flag;
  int correction_order;
  int *coordination;

 private:
  int check_corrections(int);

  char *id_fix_coord;
  class FixStore *fix_coord;

  int solid_flag;
  class ComputeRHEOSolids *compute_solids;

  int kernel_type, N2min, nmax, dim, Mdim, ncor;
  double cut, cutsq, cutinv, h, ih, ihsq, pre_w, pre_wp;
  class NeighList *list;

  int nstored_w, nstored_wp, max_nstored;
  double *stored_w;
  double **stored_wp;
  std::unordered_map<long long int, int> locations_w;
  std::unordered_map<long long int, int> locations_wp;

  int gsl_error_flag;
  std::unordered_set<tagint> gsl_error_tags;

  double ***C;
  double *C0;
  int index_coord;

  //double calc_dw_quintic(int,int,double,double,double,double,double *,double *);
  double calc_w_crk0(int,int,double,double,double,double);
  double calc_w_crk1(int,int,double,double,double,double);
  double calc_w_crk2(int,int,double,double,double,double);
  void calc_dw_crk1(int,int,double,double,double,double,double *);
  void calc_dw_crk2(int,int,double,double,double,double,double *);
  void calc_d2w_crk2(int,int,double,double,double,double,double *);

  void grow_memory();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Could not find compute coord/atom compute ID

UNDOCUMENTED

E: Compute coord/atom compute ID is not orientorder/atom

UNDOCUMENTED

E: Compute coord/atom threshold not between -1 and 1

UNDOCUMENTED

E: Invalid cstyle in compute coord/atom

UNDOCUMENTED

E: Compute coord/atom requires components option in compute orientorder/atom

UNDOCUMENTED

E: Compute coord/atom requires a pair style be defined

Self-explanatory.

E: Compute coord/atom cutoff is longer than pairwise cutoff

Cannot compute coordination at distances longer than the pair cutoff,
since those atoms are not in the neighbor list.

*/
