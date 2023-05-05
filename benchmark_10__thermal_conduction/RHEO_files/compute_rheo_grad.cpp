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

#include "compute_rheo_grad.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "compute_rheo_kernel.h"
#include "compute_rheo_solids.h"
#include "fix_rheo.h"
#include "fix_store.h"
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
#include "utils.h"

using namespace LAMMPS_NS;
enum{COMMGRAD, COMMFIELD};

/* ---------------------------------------------------------------------- */

ComputeRHEOGrad::ComputeRHEOGrad(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  id_fix_gradv(nullptr), id_fix_gradr(nullptr), id_fix_gradt(nullptr), id_fix_gradn(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal compute rheo/grad command");

  cut = utils::numeric(FLERR,arg[3],false,lmp);

  velocity_flag = temperature_flag = rho_flag = eta_flag = 0;
  for (int iarg = 4; iarg < narg; iarg ++) {
    if (strcmp(arg[iarg],"velocity") == 0) velocity_flag = 1;
    else if (strcmp(arg[iarg],"rho") == 0) rho_flag = 1;
    else if (strcmp(arg[iarg],"temperature") == 0) temperature_flag = 1;
    else if (strcmp(arg[iarg],"viscosity") == 0) eta_flag = 1;
    else error->all(FLERR, "Illegal compute rheo/grad command");
  }

  cutsq = cut*cut;
  dim = domain->dimension;

  nmax = 0;
  ncomm_grad = 0;
  ncomm_field = 0;
  comm_reverse = 0;

  if (velocity_flag) {
    ncomm_grad += dim*dim;
    ncomm_field += dim;
    comm_reverse += dim*dim;
  }

  if (rho_flag) {
    ncomm_grad += dim;
    ncomm_field += 1;
    comm_reverse += dim;
  }

  if (temperature_flag) {
    ncomm_grad += dim;
    ncomm_field += 1;
    comm_reverse += dim;
  }

  if (eta_flag) {
    ncomm_grad += dim;
    comm_reverse += dim;
  }

  comm_forward = ncomm_grad;

  fix_gradv = nullptr;
  fix_gradr = nullptr;
  fix_gradt = nullptr;
  fix_gradn = nullptr;

  // new id = fix-ID + FIX_STORE_ATTRIBUTE
  // new fix group = group for this fix

  id_fix_gradv = nullptr;
  id_fix_gradr = nullptr;
  id_fix_gradt = nullptr;
  id_fix_gradn = nullptr;

  if (velocity_flag && !fix_gradv) {
    std::string fixcmd = id + std::string("_gradv");
    id_fix_gradv = new char[fixcmd.size()+1];
    strcpy(id_fix_gradv,fixcmd.c_str());
    fixcmd += fmt::format(" all STORE peratom 0 {}", dim*dim);
    modify->add_fix(fixcmd);
    fix_gradv = (FixStore *) modify->fix[modify->nfix-1];
    gradv = fix_gradv->astore;
  }

  if (rho_flag && !fix_gradr) {
    std::string fixcmd = id + std::string("_gradr");
    id_fix_gradr = new char[fixcmd.size()+1];
    strcpy(id_fix_gradr,fixcmd.c_str());
    fixcmd += fmt::format(" all STORE peratom 0 {}", dim);
    modify->add_fix(fixcmd);
    fix_gradr = (FixStore *) modify->fix[modify->nfix-1];
    gradr = fix_gradr->astore;
  }

  if (temperature_flag && !fix_gradt) {
    std::string fixcmd = id + std::string("_gradt");
    id_fix_gradt = new char[fixcmd.size()+1];
    strcpy(id_fix_gradt,fixcmd.c_str());
    fixcmd += fmt::format(" all STORE peratom 0 {}", dim);
    modify->add_fix(fixcmd);
    fix_gradt = (FixStore *) modify->fix[modify->nfix-1];
    gradt = fix_gradt->astore;
  }

  if (eta_flag && !fix_gradn) {
    std::string fixcmd = id + std::string("_gradn");
    id_fix_gradn = new char[fixcmd.size()+1];
    strcpy(id_fix_gradn,fixcmd.c_str());
    fixcmd += fmt::format(" all STORE peratom 0 {}", dim);
    modify->add_fix(fixcmd);
    fix_gradn = (FixStore *) modify->fix[modify->nfix-1];
    gradn = fix_gradn->astore;
  }

  compute_solids = nullptr;
  compute_kernel = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeRHEOGrad::~ComputeRHEOGrad()
{
  if (id_fix_gradv && modify->nfix) modify->delete_fix(id_fix_gradv);
  if (id_fix_gradr && modify->nfix) modify->delete_fix(id_fix_gradr);
  if (id_fix_gradt && modify->nfix) modify->delete_fix(id_fix_gradt);
  if (id_fix_gradn && modify->nfix) modify->delete_fix(id_fix_gradn);
}


/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::init()
{
  // need an occasional full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;

  int icompute = modify->find_compute("rheo_kernel");
  if (icompute == -1) error->all(FLERR, "Using compute/rheo/grad without compute/rheo/kernel");
  compute_kernel = ((ComputeRHEOKernel *) modify->compute[icompute]);

  icompute = modify->find_compute("rheo_solids");
  if (icompute == -1) error->all(FLERR, "Using compute/rheo/grad without compute/rheo/solids");
  compute_solids = ((ComputeRHEOSolids *) modify->compute[icompute]);
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::compute_peratom()
{
  int i, j, k, ii, jj, jnum, itype, jtype, a, b;
  double xtmp, ytmp, ztmp, delx, dely, delz, Ti, Tj;
  double rsq, imass, jmass;
  int *jlist;
  double wp;
  double rhoi, rhoj, Voli, Volj, rhoij, Tij, etai, etaj;
  double vi[3], vj[3], vij[3];
  double *dWij, *dWji;

  // neighbor list variables
  int inum, *ilist, *numneigh, **firstneigh;
  int nlocal = atom->nlocal;

  double **x = atom->x;
  double **v = atom->v;
  double *rho = atom->rho;
  double *temp = atom->temp;
  double *eta = atom->viscosity;
  int *phase = atom->phase;
  int *type = atom->type;
  double *mass = atom->mass;
  int newton = force->newton;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    if (velocity_flag) {
      fix_gradv->grow_arrays(nmax);
      gradv = fix_gradv->astore;
    }
    if (rho_flag) {
      fix_gradr->grow_arrays(nmax);
      gradr = fix_gradr->astore;
    }
    if (temperature_flag) {
      fix_gradt->grow_arrays(nmax);
      gradt = fix_gradt->astore;
    }
    if (eta_flag) {
      fix_gradn->grow_arrays(nmax);
      gradn = fix_gradn->astore;
    }
  }

  // initialize arrays
  for (i = 0; i < nmax; i++) {
    if (velocity_flag) {
      for (k = 0; k < dim*dim; k++)
        gradv[i][k] = 0.0;
    }
    if (rho_flag) {
      for (k = 0; k < dim; k++)
        gradr[i][k] = 0.0;
    }
    if (temperature_flag) {
      for (k = 0; k < dim; k++)
        gradt[i][k] = 0.0;
    }
    if (eta_flag) {
      for (k = 0; k < dim; k++)
        gradn[i][k] = 0.0;
    }
  }

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    Ti = temp[i];
    etai = eta[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      jtype = type[j];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutsq) {
        for (a = 0; a < dim; a ++) {
          vi[a] = v[i][a];
          vj[a] = v[j][a];
        }

        rhoi = rho[i];
        rhoj = rho[j];
        // Add corrections for walls
        if (phase[i] <= FixRHEO::FLUID_MAX && phase[j] > FixRHEO::FLUID_MAX) {
          compute_solids->correct_v(v[i], v[j], vi, i, j);
          rhoj = compute_solids->correct_rho(j,i);
        } else if (phase[i] > FixRHEO::FLUID_MAX && phase[j] <= FixRHEO::FLUID_MAX) {
          compute_solids->correct_v(v[j], v[i], vj, j, i);
          rhoi = compute_solids->correct_rho(i,j);
        } else if (phase[i] > FixRHEO::FLUID_MAX && phase[j] > FixRHEO::FLUID_MAX) {
          rhoi = 1.0;
          rhoj = 1.0;
        }

        Tj = temp[j];
        etaj = eta[j];
        Volj = mass[jtype]/rhoj;
        Voli = mass[itype]/rhoi;

        vij[0] = vi[0] - vj[0];
        vij[1] = vi[1] - vj[1];
        vij[2] = vi[2] - vj[2];
        rhoij = rhoi-rhoj;
        Tij = Ti-Tj;

        wp = compute_kernel->calc_dw(i, j, delx, dely, delz, sqrt(rsq));
        dWij = compute_kernel->dWij;
        dWji = compute_kernel->dWji;

        for (a = 0; a < dim; a++) {
          for (b = 0; b < dim; b++) {
            if (velocity_flag) // uxx uxy uxz uyx uyy uyz uzx uzy uzz
              gradv[i][a*dim+b] -= vij[a]*Volj*dWij[b];
          }

          if (rho_flag) // P,x  P,y  P,z
            gradr[i][a] -= rhoij*Volj*dWij[a];

          if (temperature_flag) // T,x  T,y  T,z
            gradt[i][a] -= Tij*Volj*dWij[a];

          if (eta_flag) // T,x  T,y  T,z
            gradn[i][a] -= (etai-etaj)*Volj*dWij[a];
        }

        if (newton || j < nlocal) {
          for (a = 0; a < dim; a++) {
            for (b = 0; b < dim; b++) {
              if (velocity_flag)
                gradv[j][a*dim+b] += vij[a]*Voli*dWji[b];
            }

            if (rho_flag)
              gradr[j][a] += rhoij*Voli*dWji[a];

            if (temperature_flag)
              gradt[j][a] += Tij*Voli*dWji[a];

            if (eta_flag) // T,x  T,y  T,z
              gradn[j][a] += (etai-etaj)*Voli*dWji[a];
          }
        }
      }
    }
  }

  if (newton) comm->reverse_comm_compute(this);
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOGrad::modify_param(int narg, char **arg)
{
  return 0;
}
/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::forward_gradients()
{
  comm_stage = COMMGRAD;
  comm_forward = ncomm_grad;
  comm->forward_comm_compute(this);
}


/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::forward_fields()
{
  comm_stage = COMMFIELD;
  comm_forward = ncomm_field;
  comm->forward_comm_compute(this);
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOGrad::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m;
  double * rho = atom->rho;
  double * temp = atom->temp;
  double * eta = atom->viscosity;
  double ** v = atom->v;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    if (comm_stage == COMMGRAD) {

      if (velocity_flag){
        for (k = 0; k < dim*dim; k++)
          buf[m++] = gradv[j][k];
      }

      if (rho_flag) {
        for (k = 0; k < dim; k++)
          buf[m++] = gradr[j][k];
      }

      if (temperature_flag) {
        for (k = 0; k < dim; k++)
          buf[m++] = gradt[j][k];
      }

      if (eta_flag){
        for (k = 0; k < dim; k++)
          buf[m++] = gradn[j][k];
      }
    } else if (comm_stage == COMMFIELD) {

      if (velocity_flag) {
        for (k = 0; k < dim; k++)
          buf[m++] = v[j][k];
      }

      if (rho_flag) {
        buf[m++] = rho[j];
      }

      if (temperature_flag) {
        buf[m++] = temp[j];
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::unpack_forward_comm(int n, int first, double *buf)
{
  int i, k, m, last;
  double * rho = atom->rho;
  double * temp = atom->temp;
  double ** v = atom->v;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (comm_stage == COMMGRAD) {
      if (velocity_flag) {
        for (k = 0; k < dim*dim; k++)
          gradv[i][k] = buf[m++];
      }
      if (rho_flag) {
        for (k = 0; k < dim; k++)
          gradr[i][k] = buf[m++];
      }
      if (temperature_flag) {
        for (k = 0; k < dim; k++)
          gradt[i][k] = buf[m++];
      }
      if (eta_flag) {
        for (k = 0; k < dim; k++)
          gradn[i][k] = buf[m++];
      }
    } else if (comm_stage == COMMFIELD) {
      if (velocity_flag) {
        for (k = 0; k < dim; k++)
          v[i][k] = buf[m++];
      }
      if (rho_flag) {
        rho[i] = buf[m++];
      }
      if (temperature_flag) {
        temp[i] = buf[m++];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOGrad::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;
  double * rho = atom->rho;
  double * temp = atom->temp;
  double ** v = atom->v;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (velocity_flag) {
      for (k = 0; k < dim*dim; k++)
        buf[m++] = gradv[i][k];
    }
    if (rho_flag) {
      for (k = 0; k < dim; k++)
        buf[m++] = gradr[i][k];
    }
    if (temperature_flag) {
      for (k = 0; k < dim; k++)
        buf[m++] = gradt[i][k];
    }
    if (eta_flag) {
      for (k = 0; k < dim; k++)
        buf[m++] = gradn[i][k];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOGrad::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,k,j,m;
  double * rho = atom->rho;
  double * temp = atom->temp;
  double ** v = atom->v;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if (velocity_flag) {
      for (k = 0; k < dim*dim; k++)
        gradv[j][k] += buf[m++];
    }
    if (rho_flag) {
      for (k = 0; k < dim; k++)
        gradr[j][k] += buf[m++];
    }
    if (temperature_flag) {
      for (k = 0; k < dim; k++)
        gradt[j][k] += buf[m++];
    }
    if (eta_flag) {
      for (k = 0; k < dim; k++)
        gradn[j][k] += buf[m++];
    }
  }
}
