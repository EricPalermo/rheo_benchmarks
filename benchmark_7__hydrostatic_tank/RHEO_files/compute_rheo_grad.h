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

ComputeStyle(rheo/grad,ComputeRHEOGrad)

#else

#ifndef LMP_COMPUTE_RHEO_GRAD_H
#define LMP_COMPUTE_RHEO_GRAD_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeRHEOGrad : public Compute {
 public:
  ComputeRHEOGrad(class LAMMPS *, int, char **);
  ~ComputeRHEOGrad();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  int modify_param(int, char **);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void forward_gradients();
  void forward_fields();
  double **gradv;
  double **gradr;
  double **gradt;
  double **gradn; //viscosity
  //double *divrdv;
  //double **divvdv;
  int stage;

 private:
  int nmax, dim, comm_stage;
  int ncomm_grad, ncomm_field;
  double cut, cutsq;
  class NeighList *list;

  int flags;
  class ComputeRHEOKernel *compute_kernel;
  class ComputeRHEOSolids *compute_solids;
  char *id_fix_gradv,*id_fix_gradr,*id_fix_gradt, *id_fix_gradn;
  class FixStore *fix_gradv, *fix_gradr, *fix_gradt, *fix_gradn;

  int velocity_flag, temperature_flag, rho_flag, eta_flag;
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
