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

#include "fix_rheo_add_heat.h"
#include "fix_rheo.h"
#include "atom.h"
#include "memory.h"
#include "modify.h"
#include "error.h"
#include "update.h"
#include "force.h"
#include "math_extra.h"

using namespace LAMMPS_NS;
using namespace FixConst;
enum {NONE, CONSTANT, TYPE, ALUMINUM};

/* ---------------------------------------------------------------------- */

FixRHEOAddHeat::FixRHEOAddHeat(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix command");
  
  heat_rate = utils::numeric(FLERR,arg[3],false,lmp);
  dynamic_group_allow = 1;
}

/* ---------------------------------------------------------------------- */

FixRHEOAddHeat::~FixRHEOAddHeat()
{  
}

/* ---------------------------------------------------------------------- */

int FixRHEOAddHeat::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;  
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEOAddHeat::init()
{
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Need to define fix rheo to use fix rheo/add/heat");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);

  if (!fix_rheo->thermal_flag) error->all(FLERR, "Need to define thermal setting in fix rheo");
  
  if (!fix_rheo->thermal_fix_defined) 
    error->all(FLERR, "Need to define fix rheo/thermal to use fix rheo/add/heat");

}

/* ---------------------------------------------------------------------- */

void FixRHEOAddHeat::post_force(int /*vflag*/)
{
  double *heat = atom->heat;
  int *mask = atom->mask;

  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      heat[i] += heat_rate;
    }    
  }
}
