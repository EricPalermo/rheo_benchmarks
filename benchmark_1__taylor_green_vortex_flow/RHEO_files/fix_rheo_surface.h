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

#ifdef FIX_CLASS

FixStyle(rheo/surface,FixRHEOSurface)

#else

#ifndef LMP_FIX_RHEO_SURFACE_H
#define LMP_FIX_RHEO_SURFACE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRHEOSurface : public Fix {
 public:
  FixRHEOSurface(class LAMMPS *, int, char **);
  ~FixRHEOSurface();
  void post_constructor();
  void init();
  void setup_pre_force(int);
  void init_list(int, class NeighList *);
  int setmask();
  void pre_force(int);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double **n_surface;
  double **gradC;

 private:
  double cut, cutsq, threshold;
  class NeighList *list;
  int nmax;
  double **B, *divr;
  int comm_stage;

  int index_divr;
  int index_rsurf;

  double divR_limit;
  int coord_limit;

  class FixRHEO *fix_rheo;
  class ComputeRHEOKernel *compute_kernel;
  class ComputeRHEOSolids *compute_solids;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
