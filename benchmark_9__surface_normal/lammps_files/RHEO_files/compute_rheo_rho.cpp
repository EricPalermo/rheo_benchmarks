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

#include "compute_rheo_rho.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "compute_rheo_kernel.h"
#include "fix_rheo.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
enum {COORDINATION, RHO};
/* ---------------------------------------------------------------------- */

ComputeRHEORho::ComputeRHEORho(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 5) error->all(FLERR,"Illegal compute RHEO/rho command");

  cut = utils::numeric(FLERR,arg[3],false,lmp);
  thres_coord = utils::inumeric(FLERR,arg[4],false,lmp);
  cutsq = cut*cut;

  nmax = atom->nmax;

  comm_forward = 1;
  comm_reverse = 1;

  mass_weight = 1.0;
  initial = 1;

  nmax = 0;
  coordination = nullptr;

  error->warning(FLERR,"Compute RHEO/Rho has not been updated");
}

/* ---------------------------------------------------------------------- */

ComputeRHEORho::~ComputeRHEORho()
{
  memory->destroy(coordination);
}

/* ---------------------------------------------------------------------- */

void ComputeRHEORho::init()
{
  // need an occasional full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;
  //neighbor->requests[irequest]->occasional = 1; //Anticipate needing regulalry

  int icompute = modify->find_compute("rheo_kernel");
  if (icompute == -1) error->all(FLERR, "Using compute/RHEO/rho without compute/RHEO/kernel");

  compute_kernel = ((ComputeRHEOKernel *) modify->compute[icompute]);
}

/* ---------------------------------------------------------------------- */

void ComputeRHEORho::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */


void ComputeRHEORho::compute_peratom()
{

  int nall = atom->nlocal + atom->nghost;

  if(nmax <= nall){
    nmax = nall;
    memory->destroy(coordination);
    memory->create(coordination,nmax,"compute/rheo/rho:coordination");
  }

  calc_coord();

  if(initial and thres_coord == 0){
    initial = 0;
    calc_rho();

    double rho_total = 0;
    int coord_total = 0;
    double *rho = atom->rho;
    int nlocal = atom->nlocal;
    for(int i = 0; i < nlocal; i++){
      rho_total += rho[i];
      coord_total += coordination[i];
    }

    int n_global, coord_global;
    double rho_global;
    MPI_Allreduce(&nlocal, &n_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&coord_total, &coord_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&rho_total, &rho_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    mass_weight = n_global/rho_global;
    ave_coord = coord_global/n_global;

    thres_coord  = 0.75*ave_coord;
    if(comm->me == 0) printf("Weight %g, ave coord %d\n", mass_weight, ave_coord);
  }

  calc_rho();
}

/* ---------------------------------------------------------------------- */

void ComputeRHEORho::calc_coord()
{
  int i, j, ii, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double rsq, imass;
  int *jlist;
  double w;

  // neighbor list variables
  int inum, *ilist, *numneigh, **firstneigh;
  int nlocal = atom->nlocal;

  double **x = atom->x;
  double *rho = atom->rho;
  int *type = atom->type;
  int *phase = atom->phase;
  double *mass = atom->mass;
  int newton = force->newton;

  double jmass;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  int nall = atom->nlocal + atom->nghost;

  // initialize arrays
  for (i = 0; i < nall; i++) {
    coordination[i]=0;
  }

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    //if(phase[i] != FixRHEO::FLUID) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      //if(phase[j] != FixRHEO::FLUID) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < cutsq) {
        coordination[i] += 1;
        //Add to neighbor if it's not a ghost atom
        if(j < nlocal || newton){
          coordination[j] += 1;
        }
      }
    }
  }

  comm_stage = COORDINATION;
  if (newton) comm->reverse_comm_compute(this);
  comm->forward_comm_compute(this);
}

/* ---------------------------------------------------------------------- */

void ComputeRHEORho::calc_rho()
{
  int i, j, ii, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double rsq, imass;
  int *jlist;
  double w;

  // neighbor list variables
  int inum, *ilist, *numneigh, **firstneigh;
  int nlocal = atom->nlocal;

  double **x = atom->x;
  double *rho = atom->rho;
  int *type = atom->type;
  int *phase = atom->phase;
  double *mass = atom->mass;
  int newton = force->newton;

  double jmass;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  int nall = atom->nlocal + atom->nghost;

  // initialize arrays
  for (i = 0; i < nall; i++) {
    rho[i]=0.0;
  }

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    //if(phase[i] != FixRHEO::FLUID) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    imass = mass[type[i]];
    w = compute_kernel->calc_w_quintic(i,i,0.,0.,0.,0.);
    rho[i] += w*mass[type[i]];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      //if(phase[j] != FixRHEO::FLUID) continue;

      jmass = mass[type[j]];
      delx =  xtmp - x[j][0];
      dely =  ytmp - x[j][1];
      delz =  ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < cutsq) {
        w = compute_kernel->calc_w(i, j, delx, dely, delz, sqrt(rsq));
        rho[i] += w*mass[type[i]];
        //rho[i] += mass[type[i]]*0.5*(m0[i]+m0[j])*Wij;
        if (newton || j < nlocal) {
          rho[j] += w*mass[type[j]];
          //rho[j] += mass[type[j]]*0.5*(m0[i]+m0[j])*Wji;
        }
      }
    }
  }

  comm_stage = RHO;
  if (newton) comm->reverse_comm_compute(this);

  // Scale rho by weighting factor to reach 1-ish
  // If below threshold, identify as a surface particle and scale by missing coordination
  //for(i = 0; i < nlocal; i++){
  //  rho[i] *= mass_weight;
  //  if(coordination[i] < thres_coord)
  //    rho[i] *= (ave_coord/coordination[i]);
  //}

  comm->forward_comm_compute(this);
}

/* ---------------------------------------------------------------------- */

int ComputeRHEORho::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m;
  double * rho = atom->rho;
  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    if(comm_stage == RHO){
      buf[m++] = rho[j];
    } else if(comm_stage == COORDINATION){
      buf[m++] = coordination[j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */
void ComputeRHEORho::unpack_forward_comm(int n, int first, double *buf)
{
  int i, k, m, last;
  double * rho = atom->rho;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if(comm_stage == RHO){
      rho[i] = buf[m++];
    } else if(comm_stage == COORDINATION){
      coordination[i] = (int) buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeRHEORho::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;
  double *rho = atom->rho;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if(comm_stage == RHO){
      buf[m++] = rho[i];
    } else if(comm_stage == COORDINATION){
      buf[m++] = coordination[i];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEORho::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,k,j,m;
  double *rho = atom->rho;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if(comm_stage == RHO){
      rho[j] += buf[m++];
    } else if(comm_stage == COORDINATION){
      coordination[j] += (int) buf[m++];
    }
  }
}
