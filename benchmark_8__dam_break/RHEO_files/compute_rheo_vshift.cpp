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

#include "compute_rheo_vshift.h"
#include "fix_rheo.h"
#include "compute_rheo_solids.h"
#include "compute_rheo_grad.h"
#include "compute_rheo_kernel.h"
#include "fix_rheo_surface.h"
#include <cmath>
#include <cstring>
#include "atom.h"
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

/* ---------------------------------------------------------------------- */

ComputeRHEOVShift::ComputeRHEOVShift(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal compute RHEO/VShift command");

  cut = utils::numeric(FLERR,arg[3],false,lmp);
  cutsq = cut*cut;

  comm_reverse = 3;
  surface_flag = 0;

  nmax  = atom->nmax;
  memory->create(vshift, nmax, 3, "rheo/vshift:vshift");
  array_atom = vshift;
  peratom_flag = 1;
  size_peratom_cols = 3;

  fix_rheo = nullptr;
  compute_kernel = nullptr;
  compute_grad = nullptr;
  compute_solids = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeRHEOVShift::~ComputeRHEOVShift()
{
  memory->destroy(vshift);
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOVShift::init()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;

  int flag;
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Using pair RHEO without fix RHEO");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);

  int ifix2 = modify->find_fix_by_style("rheo/surface");
  if (ifix2 != -1) {
    fix_rheo_surface = ((FixRHEOSurface *) modify->fix[ifix2]);
    surface_flag = 1;
  } else {
    surface_flag = 0;
  }

  compute_kernel = fix_rheo->compute_kernel;
  compute_grad = fix_rheo->compute_grad;
  compute_solids = fix_rheo->compute_solids;
  rhosum_flag = fix_rheo->rhosum_flag;

  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  if (nall > nmax) {
    nmax = nall;
    memory->destroy(vshift);
    memory->create(vshift, nmax, 3, "rheo/vshift:vshift");
    array_atom = vshift;
  }

  for (int i = 0; i < nall; i++) {
    vshift[i][0] = 0.0;
    vshift[i][1] = 0.0;
    vshift[i][2] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOVShift::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOVShift::compute_peratom()
{
  int i, j, a, b, ii, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, rsq, r, ir;
  double w, wp, hthird, dr, w0, w4, magvi, magvj;
  double imass, jmass, voli, volj, rhoi, rhoj;
  double dx[3];
  double vi[3] = {0};
  double vj[3] = {0};
  int dim = domain->dimension;

  // neighbor list ariables
  int *jlist;
  int inum, *ilist, *numneigh, **firstneigh;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  double **x = atom->x;
  double **v = atom->v;
  int *type = atom->type;
  int *phase = atom->phase;
  int *surface = atom->surface;
  double *rho = atom->rho;
  double *mass = atom->mass;
  int newton_pair = force->newton_pair;


  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  if (nall > nmax) {
    nmax = nall;
    memory->destroy(vshift);
    memory->create(vshift, nmax, 3, "rheo/vshift:vshift");
    array_atom = vshift;
  }

  for (i = 0; i < nall; i++) {
    vshift[i][0] = 0.0;
    vshift[i][1] = 0.0;
    vshift[i][2] = 0.0;
  }

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    imass = mass[itype];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (phase[i] != FixRHEO::FLUID && phase[j] != FixRHEO::FLUID) continue;

      dx[0] = xtmp - x[j][0];
      dx[1] = ytmp - x[j][1];
      dx[2] = ztmp - x[j][2];
      rsq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];

      if (rsq < cutsq) {
        jtype = type[j];
        jmass = mass[jtype];

        r = sqrt(rsq);
        ir = 1/r;

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

        voli = imass/rhoi;
        volj = jmass/rhoj;

        hthird = cut/3.0;
        wp = compute_kernel->calc_dw(i, j, dx[0], dx[1], dx[2], r);
        w = compute_kernel->calc_w(i, j, dx[0], dx[1], dx[2], r);
        w0 = compute_kernel->calc_w(i, j, 0, 0, 0, hthird); // dx, dy, dz irrelevant
        w4 = w*w*w*w/(w0*w0*w0*w0);
        dr = -2*hthird*(1+0.2*w4)*wp*ir;
        magvi = sqrt(vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);
        vshift[i][0] += magvi*volj*dr*dx[0];
        vshift[i][1] += magvi*volj*dr*dx[1];
        vshift[i][2] += magvi*volj*dr*dx[2];

        if (newton_pair || j < nlocal) {
          magvj = sqrt(vj[0]*vj[0]+vj[1]*vj[1]+vj[2]*vj[2]);
          vshift[j][0] -= magvj*voli*dr*dx[0];
          vshift[j][1] -= magvj*voli*dr*dx[1];
          vshift[j][2] -= magvj*voli*dr*dx[2];
        }
      }
    }
  }

  if (newton_pair) comm->reverse_comm_compute(this);
}


/* ---------------------------------------------------------------------- */

void ComputeRHEOVShift::correct_surfaces()
{
  // Have separate step to get FixRHEOSurface->pre_force() results
  if (!surface_flag) return;

  int *phase = atom->phase;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i, a, b;
  int dim = domain->dimension;
  int *surface = atom->surface;

  double **nsurf;
  nsurf = fix_rheo_surface->n_surface;
  double nx,ny,nz,vx,vy,vz;
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (surface[i] == 1 || surface[i] == 2) {
        nx = nsurf[i][0];
        ny = nsurf[i][1];
        vx = vshift[i][0];
        vy = vshift[i][1];
        vz = vshift[i][2];
        vshift[i][0] = (1-nx*nx)*vx - nx*ny*vy;
        vshift[i][1] = (1-ny*ny)*vy - nx*ny*vx;
        if (dim > 2) {
          nz = nsurf[i][2];
          vshift[i][0] -= nx*nz*vz;
          vshift[i][1] -= ny*nz*vz;
          vshift[i][2] = (1-nz*nz)*vz - nz*ny*vy - nx*nz*vx;
        } else {
          vshift[i][2] = 0.0;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOVShift::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = vshift[i][0];
    buf[m++] = vshift[i][1];
    buf[m++] = vshift[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOVShift::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    vshift[j][0] += buf[m++];
    vshift[j][1] += buf[m++];
    vshift[j][2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeRHEOVShift::memory_usage()
{
  double bytes = 3 * nmax * sizeof(double);
  return bytes;
}

