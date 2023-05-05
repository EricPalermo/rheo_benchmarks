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

PairStyle(rheo,PairRHEO)

#else

#ifndef LMP_PAIR_RHEO_H
#define LMP_PAIR_RHEO_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRHEO : public Pair {
 public:
  PairRHEO(class LAMMPS *);
  virtual ~PairRHEO();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void setup();
  virtual double init_one(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);

 protected:
  int seed, laplacian_order;
  double **cut,*rho0, *csq;
  double h, av, rho_damp;
  int pressure_style;
  double c_pressure_cubic;

  int artificial_visc_flag;
  int rho_damp_flag;
  int thermal_flag;

  void allocate();
  double calc_pressure(double,int);

  class ComputeRHEOKernel *compute_kernel;
  class ComputeRHEOGrad *compute_grad;
  class ComputeRHEOSolids *compute_solids;
  class FixRHEO *fix_rheo;
};

}

#endif
#endif
