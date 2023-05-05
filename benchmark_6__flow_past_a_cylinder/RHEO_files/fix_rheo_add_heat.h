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

FixStyle(rheo/add/heat,FixRHEOAddHeat)

#else

#ifndef LMP_FIX_RHEO_ADD_HEAT_H
#define LMP_FIX_RHEO_ADD_HEAT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRHEOAddHeat : public Fix {
 public:
  FixRHEOAddHeat(class LAMMPS *, int, char **);
  ~FixRHEOAddHeat();
  int setmask();
  void init();  
  void post_force(int);

 private:
  double heat_rate;
  class FixRHEO *fix_rheo;
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
