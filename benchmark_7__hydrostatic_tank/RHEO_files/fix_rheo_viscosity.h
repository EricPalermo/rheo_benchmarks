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

FixStyle(rheo/viscosity,FixRHEOViscosity)

#else

#ifndef LMP_FIX_RHEO_VISCOSITY_H
#define LMP_FIX_RHEO_VISCOSITY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRHEOViscosity : public Fix {
 public:
  FixRHEOViscosity(class LAMMPS *, int, char **);
  ~FixRHEOViscosity();
  virtual void setup_pre_force(int);  
  int setmask();
  void init();  
  void pre_force(int);  
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
 private:
  double *eta_type, eta, AL_a, AL_b, AL_T0, AL_Tc; 
  double  npow, K, gd0, tau0, max_tau;
  int viscosity_style;
  class FixRHEO *fix_rheo;
  class ComputeRHEOGrad *compute_grad;
  class ComputeStressAtom *compute_stress;
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
