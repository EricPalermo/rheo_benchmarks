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

#include "fix_rheo.h"
#include "compute_rheo_grad.h"
#include "compute_rheo_solids.h"
#include "compute_rheo_kernel.h"
#include "compute_rheo_rho.h"
#include "compute_rheo_vshift.h"
#include "fix_store.h"
#include <cstring>
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "modify.h"
#include "comm.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRHEO::FixRHEO(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

  thermal_flag = 0;
  rhosum_flag = 0;
  shift_flag = 0;

  if (atom->rho_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with density");
  if (atom->temp_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with temperature");
  if (atom->viscosity_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with viscosity");
  if (atom->conductivity_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with conductivity");
  if (atom->phase_flag != 1)
    error->all(FLERR,"fix rheo command requires atom_style with phase");


  if (narg < 5)
    error->all(FLERR,"Illegal number of arguments for fix rheo command");

  cut = utils::numeric(FLERR,arg[3],false,lmp);
  if (strcmp(arg[4],"Quintic") == 0) {
      kernel_type = QUINTIC;
  } else if (strcmp(arg[4],"CRK0") == 0) {
      kernel_type = CRK0;
  } else if (strcmp(arg[4],"CRK1") == 0) {
      kernel_type = CRK1;
  } else if (strcmp(arg[4],"CRK2") == 0) {
      kernel_type = CRK2;
  } else error->all(FLERR,"Unknown kernel in fix rheo");
  N2min = utils::numeric(FLERR,arg[5],false,lmp);


  int iarg = 6;
  while (iarg < narg){
    if (strcmp(arg[iarg],"shift") == 0) {
      shift_flag = 1;
    } else if (strcmp(arg[iarg],"thermal") == 0) {
      thermal_flag = 1;
    } else if (strcmp(arg[iarg],"rhosum") == 0) {
      rhosum_flag = 1;
      if(iarg+1 >= narg) error->all(FLERR,"Illegal fix command");
      surface_coordination = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 1;
    } else {
      error->all(FLERR, "Illegal fix rheo command");
    }
    iarg += 1;
  }

  time_integrate = 1;
  compute_grad = nullptr;
  compute_kernel = nullptr;
  compute_solids = nullptr;

  thermal_fix_defined = 0;
  viscosity_fix_defined = 0;
}

/* ---------------------------------------------------------------------- */

FixRHEO::~FixRHEO()
{
  if (modify->ncompute) modify->delete_compute("rheo_kernel");
  if (modify->ncompute) modify->delete_compute("rheo_grad");
  if (modify->ncompute) modify->delete_compute("rheo_solids");

  if (rhosum_flag && modify->ncompute) modify->delete_compute("rheo_rho");
  if (shift_flag && modify->ncompute) modify->delete_compute("rheo_vshift");
}

/* ---------------------------------------------------------------------- */

void FixRHEO::post_constructor()
{
  char gcutstr[16];
  sprintf(gcutstr,"%g",cut);

  char dN2minstr[16];
  sprintf(dN2minstr,"%d",N2min);

  char **newarg = new char*[6];
  newarg[0] = (char *) "rheo_kernel";
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "rheo/kernel";
  if (kernel_type == QUINTIC) newarg[3] = (char *) "Quintic";
  if (kernel_type == CRK0) newarg[3] = (char *) "CRK0";
  if (kernel_type == CRK1) newarg[3] = (char *) "CRK1";
  if (kernel_type == CRK2) newarg[3] = (char *) "CRK2";
  newarg[4] = dN2minstr;
  newarg[5] = gcutstr;
  modify->add_compute(6,newarg);
  delete [] newarg;
  compute_kernel = ((ComputeRHEOKernel *) modify->compute[modify->ncompute-1]);

  int ngrad = 3;
  if (thermal_flag) ngrad += 1;
  int narg = 0;

  newarg = new char*[4 + ngrad];
  newarg[narg++] = (char *) "rheo_grad";
  newarg[narg++] = (char *) "all";
  newarg[narg++] = (char *) "rheo/grad";
  newarg[narg++] = gcutstr;
  newarg[narg++] = (char *) "velocity";
  newarg[narg++] = (char *) "rho";
  newarg[narg++] = (char *) "viscosity";
  if (thermal_flag) newarg[narg++] = (char *) "temperature";
  modify->add_compute(narg,newarg);
  delete [] newarg;
  compute_grad = ((ComputeRHEOGrad *) modify->compute[modify->ncompute-1]);

  newarg = new char*[4];
  newarg[0] = (char *) "rheo_solids";
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "rheo/solids";
  newarg[3] = gcutstr;
  modify->add_compute(4,newarg);
  delete [] newarg;
  compute_solids = ((ComputeRHEOSolids *) modify->compute[modify->ncompute-1]);


  if (rhosum_flag) {
    char coordstr[16];
    sprintf(coordstr,"%d",surface_coordination);

    newarg = new char*[5];
    newarg[0] = (char *) "rheo_rho";
    newarg[1] = (char *) "all";
    newarg[2] = (char *) "rheo/rho";
    newarg[3] = gcutstr;
    newarg[4] = coordstr;
    modify->add_compute(5,newarg);
    delete [] newarg;
    compute_rho = ((ComputeRHEORho *) modify->compute[modify->ncompute-1]);
  }

  if (shift_flag) {
    newarg = new char*[4];
    newarg[0] = (char *) "rheo_vshift";
    newarg[1] = (char *) "all";
    newarg[2] = (char *) "rheo/vshift";
    newarg[3] = gcutstr;
    modify->add_compute(4,newarg);
    delete [] newarg;
    compute_vshift = ((ComputeRHEOVShift *) modify->compute[modify->ncompute-1]);
  }
}

/* ---------------------------------------------------------------------- */

int FixRHEO::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEO::init()
{
  // Run calls lmp->init() then integrate->setup() at start of run
  // LAMMPS->init() calls force(pair)->init() then modify->init()
  // Modify->init() first calls compute->init() then fix->init()
  // Integrate(verlet)->setup() calls modify->setup_pre_exchnge(), force(pair)->setup(), modify->setup_pre_force(), then modify->setup()
  // Modify->setup() first calls compute->setup() then fix->setup(), all other modify->setups only call fix->setup equivalent
  // SO all compute should have already initialized, can check to make sure necessary compute exist

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixRHEO::setup_pre_force(int /*vflag*/)
{
  // Check to confirm all accessory fixes are defined
  if (thermal_flag) {
    if(!thermal_fix_defined)
      error->all(FLERR, "Missing thermal evolution fix");
  }

  if (!viscosity_fix_defined)
    error->all(FLERR, "Must specify viscosity fix");

  // Reset to zero for next run
  thermal_fix_defined = 0;
  viscosity_fix_defined = 0;

  pre_force(0);
}


/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixRHEO::initial_integrate(int /*vflag*/)
{
  // update v and x and rho of atoms in group
  int i, a, b;
  double dtfm, divu;
  int dim = domain->dimension;

  int *phase = atom->phase;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rho = atom->rho;
  double *temp = atom->temp;
  double *drho = atom->drho;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int rmass_flag = atom->rmass_flag;

  double **gradr = compute_grad->gradr;
  double **gradv = compute_grad->gradv;
  double **gradt = compute_grad->gradt;

  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  //Density Half-step

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      if (phase[i] == FLUID_NO_FORCE) continue;

      v[i][0] += dtfm*f[i][0];
      v[i][1] += dtfm*f[i][1];
      v[i][2] += dtfm*f[i][2];

      //if (phase[i] == FLUID && shift_flag) {
      //  for (a = 0; a < dim; a++)
      //    v[i][a] += dtf*divvdv[i][a];
      //}
    }
  }

  //Density Half-step
  compute_grad->forward_fields(); // also forwards v and rho for chi
  compute_solids->store_forces(); // Need to save, wiped in exchange
  compute_solids->compute_peratom();
  compute_grad->stage = 1;
  compute_grad->compute_peratom();

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      //update position
      for (a = 0; a < dim; a++) {
        x[i][a] += dtv*v[i][a];
      }

      if (phase[i] == FLUID_NO_FORCE) continue;

      //Compute div(u) for density
      if (!rhosum_flag && (phase[i] <= FLUID_MAX)) {
        divu = 0;
        for (a = 0; a < dim; a++) {
          divu += gradv[i][a*(1+dim)];
        }
        rho[i] += dtf*(drho[i] - rho[i]*divu);
      }
    }
  }

  if (shift_flag) {
    compute_vshift->correct_surfaces();
    double **vshift = compute_vshift->array_atom;
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        //update position
        for (a = 0; a < dim; a++) {
          if (phase[i] == FLUID) {
            x[i][a] += dtv*vshift[i][a];
            for (b = 0; b < dim; b++) {
              v[i][a] += dtv*vshift[i][b]*gradv[i][a*dim + b];
            }
          }
        }


        //Compute div(u) for density
        if (!rhosum_flag && phase[i] == FLUID) {
          for (a = 0; a < dim; a++) {
            rho[i] += dtv*vshift[i][a]*gradr[i][a];
          }
        }
        //interpolate temperature
        if (thermal_flag && phase[i] == FLUID) {
          for (a = 0; a < dim; a++) {
            temp[i] += dtv*vshift[i][a]*gradt[i][a];
          }
        }
      }
    }
  }
}


/* ---------------------------------------------------------------------- */

void FixRHEO::pre_force(int /*vflag*/)
{
  if (rhosum_flag)
    compute_rho->compute_peratom();

  compute_grad->forward_fields(); // also forwards v and rho for chi
  compute_kernel->compute_peratom();
  compute_solids->compute_peratom();

  //compute_grad->stage = 0; does nothing now
  compute_grad->compute_peratom();
  compute_grad->forward_gradients();

  if (shift_flag)
    compute_vshift->compute_peratom();

  int *phase = atom->phase;
  for (int i = 0; i < atom->nlocal+atom->nghost; i++) {
    // Reset phase before force calculation, may be turned off
    // by other fixes/pair styles
    if (phase[i] <= FLUID_MAX) phase[i] = FLUID;
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::final_integrate() {
  // update v and rho of atoms in group
  int *phase = atom->phase;
  double **gradv = compute_grad->gradv;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  double *rho = atom->rho;
  double *drho = atom->drho;
  int *type = atom->type;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;
  double dtfm, divu;
  double *rmass = atom->rmass;
  int rmass_flag = atom->rmass_flag;
  int i, a;

  int dim = domain->dimension;


  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (phase[i] == FLUID_NO_FORCE) continue;

      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      for (a = 0; a < dim; a++) {
        v[i][a] += dtfm*f[i][a];
      }

      //Compute div(u) for density
      if (!rhosum_flag && (phase[i] <= FLUID_MAX)) {
        divu = 0;
        for (a = 0; a < dim; a++) {
          divu += gradv[i][a*(1+dim)];
        }
        rho[i] += dtf*(drho[i] - rho[i]*divu);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRHEO::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}


/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void FixRHEO::force_clear()
{
  size_t nbytes;
  int nlocal = atom->nlocal;
  nbytes = sizeof(double) * atom->nfirst;
  if (nbytes) {
    memset(&atom->f[0][0],0,3*nbytes);
    memset(&atom->drho[0],0,nbytes);
    memset(&atom->heat[0],0,nbytes);
  }
  if (force->newton) {
    nbytes = sizeof(double) * atom->nghost;
    if (nbytes) {
      memset(&atom->f[nlocal][0],0,3*nbytes);
      memset(&atom->drho[nlocal],0,nbytes);
      memset(&atom->heat[nlocal],0,nbytes);
    }
  }
}
