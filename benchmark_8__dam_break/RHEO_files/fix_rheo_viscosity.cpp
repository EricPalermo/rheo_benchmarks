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

#include "fix_rheo_viscosity.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "fix_rheo.h"
#include "compute_rheo_grad.h"
#include "compute_stress_atom.h"
#include "atom.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "domain.h"
#include "comm.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;
enum {NONE, CONSTANT, TYPE, POWER, BINGHAM, BINGHAM2, T_AL};

/* ---------------------------------------------------------------------- */

FixRHEOViscosity::FixRHEOViscosity(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix command");

  viscosity_style = NONE;

  eta_type = nullptr;
  comm_forward = 1;

  int ntypes = atom->ntypes;
  int iarg = 3;
  if (strcmp(arg[iarg],"constant") == 0) {
    if (iarg+1 >= narg) error->all(FLERR,"Illegal fix command");
    viscosity_style = CONSTANT;
    eta = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    if(eta < 0.0) error->all(FLERR,"Illegal fix command");
    iarg += 1;
  } else if (strcmp(arg[iarg],"type") == 0) {
    if(iarg+ntypes >= narg) error->all(FLERR,"Illegal fix command");
    viscosity_style = TYPE;
    memory->create(eta_type,ntypes+1,"rheo_thermal:eta_type");
    for (int i = 1; i <= ntypes; i++) {
      eta_type[i] = utils::numeric(FLERR,arg[iarg+1+i],false,lmp);
      if (eta_type[i] < 0.0) error->all(FLERR,"Illegal fix command");
    }
    iarg += ntypes;
  } else if (strcmp(arg[iarg],"power") == 0) {
    if (iarg+4 >= narg) error->all(FLERR,"Illegal fix command");
    viscosity_style = POWER;
    eta = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    gd0 = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    K = utils::numeric(FLERR,arg[iarg+3],false,lmp);
    npow = utils::numeric(FLERR,arg[iarg+4],false,lmp);
    tau0 = eta*gd0 - K*pow(gd0,npow);
    if (eta < 0.0) error->all(FLERR,"Illegal fix command");
    iarg += 5;
  } else if (strcmp(arg[iarg],"bingham") == 0) {
    if (iarg+4 >= narg) error->all(FLERR,"Illegal fix command");
    viscosity_style = BINGHAM;
    eta = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    gd0 = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    K = utils::numeric(FLERR,arg[iarg+3],false,lmp);
    npow = utils::numeric(FLERR,arg[iarg+4],false,lmp);
    tau0 = eta*gd0 - K*pow(gd0,npow);
    if (eta < 0.0) error->all(FLERR,"Illegal fix command");
    comm_forward += 1;
    iarg += 5;
  } else if (strcmp(arg[iarg],"bingham2") == 0) {
    if (iarg+5 >= narg) error->all(FLERR,"Illegal fix command");
    viscosity_style = BINGHAM2;
    eta = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    gd0 = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    K = utils::numeric(FLERR,arg[iarg+3],false,lmp);
    npow = utils::numeric(FLERR,arg[iarg+4],false,lmp);
    max_tau = utils::numeric(FLERR,arg[iarg+5],false,lmp);
    tau0 = eta*gd0 - K*pow(gd0,npow);
    if (eta < 0.0) error->all(FLERR,"Illegal fix command");
    comm_forward += 1;
    iarg += 6;
  } else if (strcmp(arg[iarg],"aluminum") == 0) {
    if (iarg+5 >= narg) error->all(FLERR,"Illegal fix command");
    viscosity_style = T_AL;
    eta = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    AL_a = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    AL_b = utils::numeric(FLERR,arg[iarg+3],false,lmp);
    AL_T0 = utils::numeric(FLERR,arg[iarg+4],false,lmp);
    AL_Tc = utils::numeric(FLERR,arg[iarg+5],false,lmp);
    if (eta < 0.0) error->all(FLERR,"Illegal fix command");
    iarg += 6;
  } else {
    error->all(FLERR,"Illegal fix command");
  }

  if (viscosity_style == NONE)
    error->all(FLERR,"Illegal fix command");

  compute_stress = nullptr;
  if (viscosity_style == BINGHAM2) {
    char **newarg = new char*[4];
    newarg[0] = (char *) "rheo_stress";
    newarg[1] = (char *) "all";
    newarg[2] = (char *) "stress/atom";
    newarg[3] = (char *) "NULL";
    modify->add_compute(4,newarg);
    delete [] newarg;
    compute_stress = ((ComputeStressAtom *) modify->compute[modify->ncompute-1]);
  }
}

/* ---------------------------------------------------------------------- */

FixRHEOViscosity::~FixRHEOViscosity()
{
  // If fix rheo is still defined, remove any set flags
  if(fix_rheo){
    //fix_rheo->viscosity_fix_defined = 0;
  }

  memory->destroy(eta_type);
  if (compute_stress) modify->delete_compute("rheo_stress");
}

/* ---------------------------------------------------------------------- */

int FixRHEOViscosity::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEOViscosity::init()
{
  int flag;
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Need to define fix rheo to use fix rheo/viscosity");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);

  //if(fix_rheo->viscosity_fix_defined)
  //  error->all(FLERR, "Cannot define two rheo viscosity fixes");
  fix_rheo->viscosity_fix_defined = 1;

  if (viscosity_style == T_AL && (! fix_rheo->thermal_flag))
    error->all(FLERR, "Need to define fix rheo/thermal to use temperature based viscosities");

  //Assign compute grad from fix_rheo
  compute_grad = fix_rheo->compute_grad;
}


/* ---------------------------------------------------------------------- */

void FixRHEOViscosity::setup_pre_force(int /*vflag*/)
{
  pre_force(0);
}

/* ---------------------------------------------------------------------- */

void FixRHEOViscosity::pre_force(int /*vflag*/)
{
  int *type = atom->type;
  double *viscosity = atom->viscosity;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int dim = domain->dimension;
  double tmp, Ti;
  double gdot;
  double **gradv = compute_grad->gradv;
  double **stress;
  double s;
  int *phase = atom->phase;
  int i, a, b;
  double max_s = 0.0;

  if (viscosity_style == BINGHAM2) {
    if (update->setupflag == 0 && update->ntimestep != 0)
      compute_stress->compute_peratom();
    stress = compute_stress->array_atom;
    compute_stress->addstep(update->ntimestep+1);
  }

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (viscosity_style == CONSTANT) {
        viscosity[i] = eta;
      } else if (viscosity_style == TYPE) {
        viscosity[i] = eta_type[type[i]];
      } else if (viscosity_style == POWER) {
        gdot = 0.0;
        for (a = 0; a < dim; a++) {
          for (b = a; b < dim; b++) {
            tmp = gradv[i][a*dim+b] + gradv[i][b*dim+a];
            tmp = tmp*tmp;
            if (a == b) tmp *= 0.5;
            gdot += tmp;
          }
        }
        gdot = sqrt(gdot);
        if (gdot <= gd0) {
          viscosity[i] = eta;
        } else {
          viscosity[i] = K*pow(gdot,npow-1) + tau0/gdot;
        }
      } else if (viscosity_style == BINGHAM) {
        gdot = 0.0;
        for (a = 0; a < dim; a++) {
          for (b = a; b < dim; b++) {
            tmp = gradv[i][a*dim+b] + gradv[i][b*dim+a];
            tmp = tmp*tmp;
            if (a == b) tmp *= 0.5;
            gdot += tmp;
          }
        }
        gdot = sqrt(gdot);
        if (gdot <= gd0) {
          viscosity[i] = eta;
          phase[i] = FixRHEO::REACTIVE;
        } else {
          viscosity[i] = K*pow(gdot,npow-1) + tau0/gdot;
          phase[i] = FixRHEO::FLUID;
        }
      } else if (viscosity_style == BINGHAM2) {
        gdot = 0.0;
        for (a = 0; a < dim; a++) {
          for (b = a; b < dim; b++) {
            tmp = gradv[i][a*dim+b] + gradv[i][b*dim+a];
            tmp = tmp*tmp;
            if (a == b) tmp *= 0.5;
            gdot += tmp;
          }
        }
        if (update->setupflag == 0 && update->ntimestep != 0) {
          s = (stress[i][0]-stress[i][1])*(stress[i][0]-stress[i][1]);
          s += (stress[i][1]-stress[i][2])*(stress[i][1]-stress[i][2]);
          s += (stress[i][2]-stress[i][0])*(stress[i][2]-stress[i][0]);
          s *= (1.0/6.0);
          s += stress[i][3]*stress[i][3];
          s += stress[i][4]*stress[i][4];
          s += stress[i][5]*stress[i][5];
          s = sqrt(s);
          if (s > max_s) max_s = s;
        }
        gdot = sqrt(gdot);
        if (gdot <= gd0) {
          viscosity[i] = eta;
          phase[i] = FixRHEO::REACTIVE;
          if (update->setupflag == 0 && update->ntimestep != 0)
            if (s > max_tau)
              phase[i] = FixRHEO::FLUID;
        } else {
          viscosity[i] = K*pow(gdot,npow-1) + tau0/gdot;
          phase[i] = FixRHEO::FLUID;
        }
      } else if (viscosity_style == T_AL) {
        Ti = atom->temp[i];
        if (Ti > AL_Tc) {
          viscosity[i] = eta*exp(-AL_a/(1+exp(AL_b*(AL_T0-Ti))));
        } else {
          viscosity[i] = eta*exp(-AL_a/(1+exp(AL_b*(AL_T0-AL_Tc))));
        }
      }
    }
  }

//  if(comm->me == 0 and (update->setupflag == 0 && update->ntimestep != 0) ) printf("maxs = %g (%g, %g) vs %g %g\n", max_s, s, stress[0][0], max_tau, tau0);

  comm->forward_comm_fix(this);
}

/* ---------------------------------------------------------------------- */

int FixRHEOViscosity::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m;
  int *phase = atom->phase;
  double *viscosity = atom->viscosity;
  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = viscosity[j];
    if (viscosity_style == BINGHAM || viscosity_style == BINGHAM2)
      buf[m++] = phase[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRHEOViscosity::unpack_forward_comm(int n, int first, double *buf)
{
  int i, k, m, last;
  int *phase = atom->phase;
  double *viscosity = atom->viscosity;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    viscosity[i] = buf[m++];
    if (viscosity_style == BINGHAM || viscosity_style == BINGHAM2)
      phase[i] = buf[m++];
  }
}
