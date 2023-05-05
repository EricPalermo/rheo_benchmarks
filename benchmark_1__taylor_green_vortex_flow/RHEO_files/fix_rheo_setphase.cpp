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

#include "fix_rheo_setphase.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRHEOSetPhase::FixRHEOSetPhase(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  idregion(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal fix setforce command");

  dynamic_group_allow = 1;
  global_freq = 1;

  phase_value = utils::inumeric(FLERR,arg[3],false,lmp);

  // optional args

  iregion = -1;
  idregion = nullptr;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix setforce command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix setforce does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix setforce command");
  }
}

/* ---------------------------------------------------------------------- */

FixRHEOSetPhase::~FixRHEOSetPhase()
{
  if (copymode) return;

  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

int FixRHEOSetPhase::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEOSetPhase::init()
{
  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setforce does not exist");
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEOSetPhase::setup(int vflag)
{
  post_force(vflag);
}


/* ---------------------------------------------------------------------- */

void FixRHEOSetPhase::post_force(int /*vflag*/)
{
  double **x = atom->x;
  int *phase = atom->phase;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // update region if necessary

  Region *region = nullptr;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
      phase[i] = phase_value;
    }
}
