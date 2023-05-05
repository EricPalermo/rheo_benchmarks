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

PairStyle(rheo/freeze,PairRHEOFreeze)

#else

#ifndef LMP_PAIR_RHEO_FREEZE_H
#define LMP_PAIR_RHEO_FREEZE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRHEOFreeze : public Pair {
 public:
  PairRHEOFreeze(class LAMMPS *);
  virtual ~PairRHEOFreeze();
  virtual void compute(int, int);
  void settings(int, char **);  
  void coeff(int, char **);
  void init_style();  
  void setup();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void add_bonds(int);
  
 protected:
  double **cut,**cutbond,**cutbsq,**k, **sigma, **eps, **gamma;
  double **tscale, **tcrit, **texpand;

  void allocate();
  void transfer_history(double*, double*);
  
  int expand_flag;
  
  int size_history;
  int index_nb;
  int *dbond;
  int nmax;  

  class FixDummy *fix_dummy;
  class FixNeighHistory *fix_history;  
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

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair granular requires atom attributes radius, rmass

The atom style defined does not have these attributes.

E: Pair granular requires ghost atoms store velocity

Use the comm_modify vel yes command to enable this.

E: Could not find pair fix neigh history ID

UNDOCUMENTED

U: Pair granular with shear history requires newton pair off

This is a current restriction of the implementation of pair
granular styles with history.

U: Could not find pair fix ID

A fix is created internally by the pair style to store shear
history information.  You cannot delete it.

*/
