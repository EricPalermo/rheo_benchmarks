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

FixStyle(rheo,FixRHEO)

#else

#ifndef LMP_FIX_RHEO_H
#define LMP_FIX_RHEO_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRHEO : public Fix {
 public:
  FixRHEO(class LAMMPS *, int, char **);
  virtual ~FixRHEO();
  int setmask();
  virtual void post_constructor();
  virtual void init();
  virtual void setup_pre_force(int);
  virtual void pre_force(int);
  virtual void initial_integrate(int);
  virtual void final_integrate();
  void reset_dt();

  int kernel_type;
  int thermal_flag;
  int rhosum_flag;
  int shift_flag;

  int thermal_fix_defined;
  int viscosity_fix_defined;

  class ComputeRHEOGrad *compute_grad;
  class ComputeRHEOKernel *compute_kernel;
  class ComputeRHEOSolids *compute_solids;
  class ComputeRHEORho *compute_rho;
  class ComputeRHEOVShift *compute_vshift;

  // FLUID_MAX designates the maximum # of a fluid particle
  // Within fluid particles, there are some special cases
  enum {FLUID, FLUID_NO_SHIFT, FLUID_NO_FORCE, FLUID_MAX, REACTIVE, SOLID, FREEZING};
  enum {BULK, LAYER, SURFACE, SPLASH};
  enum {QUINTIC, CRK0, CRK1, CRK2};

 protected:
  double cut;
  int N2min, surface_coordination;

  double dtv,dtf;
  virtual void force_clear();
};

}

#endif
#endif
