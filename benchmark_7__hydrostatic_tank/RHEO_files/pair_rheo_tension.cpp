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

#include "pair_rheo_tension.h"
#include <cmath>
#include "atom.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "update.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRHEOTension::PairRHEOTension(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

PairRHEOTension::~PairRHEOTension() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(s);
    memory->destroy(cut);
    memory->destroy(n0);    
    memory->destroy(delta);    
    memory->destroy(a);    
    memory->destroy(b);    
  }
}

/* ---------------------------------------------------------------------- */

void PairRHEOTension::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double rsq, r, rinv, dr;
  
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *surface = atom->surface;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      // Only compute for surface particles
      if (surface[i] < 1 || surface[j] < 1) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
		dr = (b[itype][jtype]-r);
        fpair = s[itype][jtype]*(dr*dr - a[itype][jtype])/rsq;


        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (evflag)
          ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairRHEOTension::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(s, n + 1, n + 1, "pair:s");
  memory->create(n0, n + 1, n + 1, "pair:n0");
  memory->create(delta, n + 1, n + 1, "pair:delta");
  memory->create(a, n + 1, n + 1, "pair:a");
  memory->create(b, n + 1, n + 1, "pair:b");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRHEOTension::settings(int narg, char **/*arg*/) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of arguments for pair_style LLNS/tension");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairRHEOTension::coeff(int narg, char **arg) {
  if (narg != 5)
    error->all(FLERR,
        "Incorrect args for pair_style LLNS/tension coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  
  utils::bounds(FLERR,arg[0],1, atom->ntypes, ilo, ihi,error);
  utils::bounds(FLERR,arg[1],1, atom->ntypes, jlo, jhi,error);

  double s_one = utils::numeric(FLERR,arg[2],false,lmp);
  double n0_one = utils::numeric(FLERR,arg[3],false,lmp);
  double cut_one = utils::numeric(FLERR,arg[4],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      s[i][j] = s_one;
      n0[i][j] = n0_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairRHEOTension::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error->all(FLERR,"All pair LLNS/tension coeffs are not set");
  }
  int d = domain->dimension;
  double h = cut[i][j]/3.0;  
  delta[i][j] = 1.0/pow(n0[i][j], 1.0/d);
  a[i][j] = 0.25*pow(cut[i][j] - delta[i][j], 2.0);
  b[i][j] = 0.5*(cut[i][j] + delta[i][j]);

  cut[j][i] = cut[i][j];
  s[j][i] = s[i][j];
  n0[j][i] = n0[i][j];
  delta[j][i] = delta[i][j];
  a[j][i] = a[i][j];
  b[j][i] = b[i][j];
  
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairRHEOTension::single(int /*i*/, int /*j*/, int /*itype*/, int /*jtype*/,
    double /*rsq*/, double /*factor_coul*/, double /*factor_lj*/, double &fforce) {
  fforce = 0.0;

  return 0.0;
}

