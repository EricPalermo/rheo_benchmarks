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

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_cblas.h>
#include "compute_rheo_kernel.h"
#include "compute_rheo_solids.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "fix_rheo.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "pair.h"
#include "math_extra.h"
#include "utils.h"

using namespace LAMMPS_NS;
enum {QUINTIC, CRK0, CRK1, CRK2};
#define DELTA 2000

/* ---------------------------------------------------------------------- */

ComputeRHEOKernel::ComputeRHEOKernel(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  C(nullptr), stored_w(nullptr), stored_wp(nullptr), coordination(nullptr)
{
  if (narg != 6) error->all(FLERR,"Illegal compute rheo/kernel command");

  if (strcmp(arg[3],"Quintic") == 0) {
    kernel_type = QUINTIC;
    correction_order = -1;
  } else if (strcmp(arg[3],"CRK0") == 0) {
    kernel_type = CRK0;
    correction_order = 0;
  } else if (strcmp(arg[3],"CRK1") == 0) {
    kernel_type = CRK1;
    correction_order = 1;
  } else if (strcmp(arg[3],"CRK2") == 0) {
    kernel_type = CRK2;
    correction_order = 2;
  } else error->all(FLERR,"Unknown kernel in compute rheo/kernel");

  N2min = utils::inumeric(FLERR,arg[4],false,lmp);
  cut = utils::numeric(FLERR,arg[5],false,lmp);

  cutsq = cut*cut;
  cutinv = 1.0/cut;
  h = cut/3.0;
  ih = 1.0/h;
  ihsq = ih*ih;

  dim = domain->dimension;

  comm_forward = 0;
  nmax = atom->nmax;

  if (kernel_type == CRK0) {
    memory->create(C0, nmax, "rheo/kernel:C0");
    comm_forward = 1;
  }

  if (kernel_type == CRK1) {
    Mdim = 1 + dim;
    ncor = 1 + dim;
    memory->create(C, nmax, ncor, Mdim, "rheo/kernel:C");
    comm_forward = ncor*Mdim;
  }

  if (kernel_type == CRK2) {
    //Polynomial basis size (up to quadratic order)
    Mdim = 1 + dim + dim*(dim+1)/2;
    //Number of sets of correction coefficients  (1 x y xx yy)  + z zz (3D)
    ncor = 1 + 2*dim;
    //variables that require forwarding
    memory->create(C, nmax, ncor, Mdim, "rheo/kernel:C");
    comm_forward = ncor*Mdim;
  }

  comm_forward += 1; // For coordination

  //Use property atom, can fix store save integers?
  char **fixarg = new char*[4];
  fixarg[0] = (char *) "PROPERTY_ATOM_RHEO_KERNEL";
  fixarg[1] = (char *) "all";
  fixarg[2] = (char *) "property/atom";
  fixarg[3] = (char *) "i_coordination";
  modify->add_fix(4,fixarg,1);

  int temp_flag;
  index_coord = atom->find_custom("coordination", temp_flag);
  if ((index_coord < 0) || (temp_flag != 0))
      error->all(FLERR, "Compute rheo/kernel can't find fix property/atom coordination");

  delete [] fixarg;
  coordination = atom->ivector[index_coord];

  if (dim == 3) {
    pre_w = 0.002652582384864922*ihsq*ih;
    pre_wp = pre_w*ih;
  } else {
    pre_w = 0.004661441847879780*ihsq;
    pre_wp = pre_w*ih;
  }

  memory_flag = 0;
  max_nstored = 0;

  gsl_error_flag = 0;
  gsl_error_tags.clear();

  compute_solids = nullptr;
  solid_flag = 0;
}

/* ---------------------------------------------------------------------- */

ComputeRHEOKernel::~ComputeRHEOKernel()
{
  if (modify->nfix) modify->delete_fix("PROPERTY_ATOM_RHEO_KERNEL");

  if (correction_order > 0)
    memory->destroy(C);
  if (correction_order == 0)
    memory->destroy(C0);

  if (stored_w != NULL) memory->destroy(stored_w);
  if (stored_wp != NULL) memory->destroy(stored_wp);

  locations_w.clear();
  locations_wp.clear();
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::init()
{
  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  int icompute = modify->find_compute("rheo_solids");
  if (icompute != -1) {
    compute_solids = ((ComputeRHEOSolids *) modify->compute[icompute]);
    solid_flag = 1;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::create_memory()
{
  if (memory_flag == 0) return;

  // Estimate number of pairs in system
  int temp = 0;
  int *numneigh = force->pair->list->numneigh;
  int inum = force->pair->list->inum;
  int *ilist = force->pair->list->ilist;
  int ii, i;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    temp += numneigh[i];
  }

  temp += DELTA;
  MPI_Allreduce(&temp,&max_nstored,1,MPI_INT,MPI_MAX,world);

  //Use initial estimate to set number of buckets
  locations_w.rehash(max_nstored);
  locations_wp.rehash(max_nstored);

  memory->create(stored_w, max_nstored,"rheo/kernel:stored_w");
  memory->create(stored_wp, max_nstored,1+dim*2,"rheo/kernel:stored_wp");
  nstored_w = 0;
  nstored_wp = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::grow_memory()
{
  max_nstored += DELTA;
  memory->grow(stored_w, max_nstored,"rheo/kernel:stored_w");
  memory->grow(stored_wp, max_nstored,1+dim*2,"rheo/kernel:stored_wp");
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::refresh_memory()
{
  nstored_w = 0;
  nstored_wp = 0;
  locations_w.clear();
  locations_wp.clear();
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOKernel::check_corrections(int i)
{
  int corrections = 1;

  if (gsl_error_flag) {
    // If there were errors, check to see if it occured for this atom
    if (gsl_error_tags.find(atom->tag[i]) != gsl_error_tags.end())
      corrections = 0;
  }

  coordination = atom->ivector[index_coord];
  if (coordination[i] < N2min) corrections = 0;

  return corrections;
}

/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_w(int i, int j, double delx, double dely, double delz, double r)
{
  double w;

  int corrections_i = check_corrections(i);
  int corrections_j = check_corrections(j);
  int corrections = corrections_i & corrections_j;

  if (memory_flag) {
    long long key;
    int v_index;
    tagint tag1 = atom->tag[i];
    tagint tag2 = atom->tag[j];

    // Use Szudzik's pairing function to define unique index for two tags
    // szudzik.com/elegantpairing.pdf
    if (tag1 > tag2) key = (long long) tag1*tag1 + tag2;
    else key = (long long) tag2*tag2 + tag1;

    if (locations_w.find(key) != locations_w.end()){
      v_index = locations_w[key];
      w = stored_w[v_index];
    } else {
      if (kernel_type == QUINTIC || !corrections) w = calc_w_quintic(i,j,delx,dely,delz,r);
      else if (kernel_type == CRK0) w = calc_w_crk0(i,j,delx,dely,delz,r);
      else if (kernel_type == CRK1) w = calc_w_crk1(i,j,delx,dely,delz,r);
      else if (kernel_type == CRK2) w = calc_w_crk2(i,j,delx,dely,delz,r);

      locations_w[key] = nstored_w;
      stored_w[nstored_w] = w;
      nstored_w ++;
      if (nstored_w >= max_nstored) grow_memory();
    }
  } else {
    if (kernel_type == QUINTIC || !corrections) w = calc_w_quintic(i,j,delx,dely,delz,r);
    else if (kernel_type == CRK0) w = calc_w_crk0(i,j,delx,dely,delz,r);
    else if (kernel_type == CRK1) w = calc_w_crk1(i,j,delx,dely,delz,r);
    else if (kernel_type == CRK2) w = calc_w_crk2(i,j,delx,dely,delz,r);
  }

  return w;
}

/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_dw(int i, int j, double delx, double dely, double delz, double r)
{
  double wp;

  int corrections_i = check_corrections(i);
  int corrections_j = check_corrections(j);

  // Calc wp and default dW's, a bit inefficient but can redo later
  wp = calc_dw_quintic(i,j,delx,dely,delz,r,dWij,dWji);
  if(kernel_type == CRK1) {
    //check if kernel correction calculated successfully. If not, revert to quintic
    if (corrections_i) calc_dw_crk1(i,j,delx,dely,delz,r,dWij);
    if (corrections_j) calc_dw_crk1(j,i,-delx,-dely,-delz,r,dWji);
  } else if(kernel_type == CRK2) {
    if (corrections_i) calc_dw_crk2(i,j,delx,dely,delz,r,dWij);
    if (corrections_j) calc_dw_crk2(j,i,-delx,-dely,-delz,r,dWji);
  }

  return wp;
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOKernel::calc_d2w(int i, int j, double delx, double dely, double delz, double r)
{
  if (correction_order < 2) error->all(FLERR, "Cannot request d2W operator without 2nd order corrections");

  int error_flag = 0;
  int corrections_i = check_corrections(i);
  int corrections_j = check_corrections(j);

  if(kernel_type == CRK2) {
    if (corrections_i)
      calc_d2w_crk2(i,j,delx,dely,delz,r,d2Wij);
    else error_flag = 1;

    if (corrections_j)
      calc_d2w_crk2(j,i,-delx,-dely,-delz,r,d2Wji);
    else error_flag = 1;
  }

  return error_flag;
}


/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_w_quintic(int i, int j, double delx, double dely, double delz, double r)
{
  double w, tmp1, tmp2, tmp3, tmp1sq, tmp2sq, tmp3sq, s;
  s = r*ih;

	if (s>3.0) {
	  w = 0.0;
	}

	if (s <= 3.0) {
	  tmp3 = 3 - s;
	  tmp3sq = tmp3*tmp3;
	  w = tmp3sq*tmp3sq*tmp3;
	}
	if (s <= 2.0) {
      tmp2 = 2 - s;
      tmp2sq = tmp2*tmp2;
      w -= 6*tmp2sq*tmp2sq*tmp2;
	}
	if (s <= 1.0) {
	  tmp1 = 1 - s;
	  tmp1sq = tmp1*tmp1;
	  w += 15*tmp1sq*tmp1sq*tmp1;
	}

  w *= pre_w;

  Wij = w;
  Wji = w;

  return w;
}

/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_dw_quintic(int i, int j, double delx, double dely, double delz, double r, double *dW1, double *dW2)
{
  double wp, tmp1, tmp2, tmp3, tmp1sq, tmp2sq, tmp3sq, s, wprinv;
  double *mass = atom->mass;
  int *type = atom->type;

  s = r*ih;

  if (s>3.0) {
    wp = 0.0;
  }
  if (s <= 3.0) {
    tmp3 = 3 - s;
    tmp3sq = tmp3*tmp3;
    wp = -5.0*tmp3sq*tmp3sq;
  }
  if (s <= 2.0) {
    tmp2 = 2 - s;
    tmp2sq = tmp2*tmp2;
    wp += 30.0*tmp2sq*tmp2sq;
  }
  if (s <= 1.0) {
    tmp1 = 1 - s;
    tmp1sq = tmp1*tmp1;
    wp -= 75.0*tmp1sq*tmp1sq;
  }

  wp *= pre_wp;
  wprinv = wp/r;
  dW1[0] = delx*wprinv;
  dW1[1] = dely*wprinv;
  dW1[2] = delz*wprinv;

  dW2[0] = -delx*wprinv;
  dW2[1] = -dely*wprinv;
  dW2[2] = -delz*wprinv;

  return wp;
}


/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_w_crk0(int i, int j, double delx, double dely, double delz, double r)
{
  double w;

  w = calc_w_quintic(i,j,delx,dely,delz,r);

  Wij = C0[i]*w;
  Wji = C0[j]*w;

  return w;
}

/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_w_crk1(int i, int j, double delx, double dely, double delz, double r)
{
  int b;
  double w, wR, dx[3], H[Mdim];

  dx[0] = delx;
  dx[1] = dely;
  dx[2] = delz;
  w = calc_w_quintic(i,j,delx,dely,delz,r);

  if (dim == 2) {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
  } else {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = dx[2]*cutinv;
  }
  Wij = 0;
  for (b = 0; b < Mdim; b++) {
    Wij += C[i][0][b]*H[b];  // C columns: 1 x y (z) xx yy (zz)
  }
  Wij *= w;

  //Now compute Wji
  H[1] *= -1;
  H[2] *= -1;
  if (dim == 3) H[3] *= -1;

  Wji = 0;
  for (b = 0; b < Mdim; b++) {
    Wji += C[j][0][b]*H[b];  // C columns: 1 x y (z) xx yy (zz)
  }
  Wji *= w;

  return w;
}

/* ---------------------------------------------------------------------- */

double ComputeRHEOKernel::calc_w_crk2(int i, int j, double delx, double dely, double delz, double r)
{
  int b;
  double w, wR, dx[3], H[Mdim];
  dx[0] = delx;
  dx[1] = dely;
  dx[2] = delz;
  w = calc_w_quintic(i,j,delx,dely,delz,r);


  if (dim == 2) {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
    H[4] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
    H[5] = dx[0]*dx[1]*cutinv*cutinv;
  } else {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = dx[2]*cutinv;
    H[4] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
    H[5] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
    H[6] = 0.5*dx[2]*dx[2]*cutinv*cutinv;
    H[7] = dx[0]*dx[1]*cutinv*cutinv;
    H[8] = dx[0]*dx[2]*cutinv*cutinv;
    H[9] = dx[1]*dx[2]*cutinv*cutinv;
  }
  Wij = 0;
  for (b = 0; b < Mdim; b++) {
    Wij += C[i][0][b]*H[b];  // C columns: 1 x y (z) xx yy (zz)
  }
  Wij *= w;

  //Now compute Wji
  H[1] *= -1;
  H[2] *= -1;
  if (dim == 3) H[3] *= -1;

  Wji = 0;
  for (b = 0; b < Mdim; b++) {
    Wji += C[j][0][b]*H[b];  // C columns: 1 x y (z) xx yy (zz)
  }
  Wji *= w;

  return w;
}


/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::calc_dw_crk1(int i, int j, double delx, double dely, double delz, double r, double *dW)
{
  int a, b;
  double w, dx[3], H[Mdim];
  dx[0] = delx;
  dx[1] = dely;
  dx[2] = delz;

  w = calc_w_quintic(i,j,delx,dely,delz,r);
  //Populate correction basis
  if (dim == 2) {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
  } else {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = dx[2]*cutinv;
  }
  // dWij[] = dWx dWy (dWz)
  //compute derivative operators
  for (a = 0; a < dim; a++) {
    dW[a] = 0.0;
    for (b = 0; b < Mdim; b++) {
      //First derivative kernels
      dW[a] += C[i][1+a][b]*H[b]; // C columns: 1 x y (z)
    }
    dW[a] *= w;
  }
}


/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::calc_dw_crk2(int i, int j, double delx, double dely, double delz, double r, double *dW)
{
  int a, b;
  double w, dx[3], H[Mdim];
  dx[0] = delx;
  dx[1] = dely;
  dx[2] = delz;

  w = calc_w_quintic(i,j,delx,dely,delz,r);

  //Populate correction basis
  if (dim == 2) {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
    H[4] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
    H[5] = dx[0]*dx[1]*cutinv*cutinv;
  } else {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = dx[2]*cutinv;
    H[4] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
    H[5] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
    H[6] = 0.5*dx[2]*dx[2]*cutinv*cutinv;
    H[7] = dx[0]*dx[1]*cutinv*cutinv;
    H[8] = dx[0]*dx[2]*cutinv*cutinv;
    H[9] = dx[1]*dx[2]*cutinv*cutinv;
  }

  // dWij[] = dWx dWy (dWz)
  //compute derivative operators
  for (a = 0; a < dim; a++) {
    dW[a] = 0.0;
    for (b = 0; b < Mdim; b++) {
      //First derivative kernels
      dW[a] += C[i][1+a][b]*H[b]; // C columns: 1 x y (z) xx yy (zz)
    }
    dW[a] *= w;
  }
}


/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::calc_d2w_crk2(int i, int j, double delx, double dely, double delz, double r, double *d2W)
{
  int a, b;
  double w, dx[3], H[Mdim];
  dx[0] = delx;
  dx[1] = dely;
  dx[2] = delz;

  w = calc_w_quintic(i,j,delx,dely,delz,r);
  //Populate correction basis
  if (dim == 2) {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
    H[4] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
    H[5] = dx[0]*dx[1]*cutinv*cutinv;
  } else {
    H[0] = 1.0;
    H[1] = dx[0]*cutinv;
    H[2] = dx[1]*cutinv;
    H[3] = dx[2]*cutinv;
    H[4] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
    H[5] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
    H[6] = 0.5*dx[2]*dx[2]*cutinv*cutinv;
    H[7] = dx[0]*dx[1]*cutinv*cutinv;
    H[8] = dx[0]*dx[2]*cutinv*cutinv;
    H[9] = dx[1]*dx[2]*cutinv*cutinv;
  }
  // dWij[] = dWx dWy (dWz)
  //compute derivative operators
  for (a = 0; a < dim; a++) {
    d2W[a] = 0.0;
    for (b = 0; b < Mdim; b++) {
      //Second derivative kernels              //0 1 2     3  4       || 2D indexing
      d2W[a] += C[i][dim+1+a][b]*H[b];         //0 1 2 (3) 4  5  (6)  || 3D indexing
    }
    d2W[a] *= w;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::compute_peratom()
{
  gsl_error_flag = 0;
  gsl_error_tags.clear();

  int i, j, ii, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double r, rinv, rsq, imass, jmass;
  double dx[3];
  double w, wp, s, vi, vj, M, vjw;
  double vjdw[3] = {0};
  bool sflag;

  double **x = atom->x;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rho = atom->rho;
  int *phase = atom->phase;
  tagint *tag = atom->tag;
  coordination = atom->ivector[index_coord];

  int inum, *ilist, *jlist, *numneigh, **firstneigh;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  if (kernel_type == QUINTIC) {

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2]; // Will be ignored in 2D

      jlist = firstneigh[i];
      jnum = numneigh[i];
      itype = type[i];
      coordination[i] = 0;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        dx[0] = xtmp - x[j][0];
        dx[1] = ytmp - x[j][1];
        dx[2] = ztmp - x[j][2];

        rsq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        r = sqrt(rsq);
        rinv = 1/r;

        if (rsq < cutsq) {
          coordination[i] += 1;
        }
      }
    }

    comm->forward_comm_compute(this);

  } else if (kernel_type == CRK0) {
    if (atom->nmax > nmax){
      nmax = atom->nmax;
      memory->destroy(C0);
      memory->create(C0, nmax, "rheo/kernel:C");
    }

    //The moment Matrix array has to be 1D to be compatible with gsl
    // Solve linear system for each atom
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2]; // Will be ignored in 2D

      jlist = firstneigh[i];
      jnum = numneigh[i];
      itype = type[i];

      //Initialize M and H to zero:
      M = 0;

      //variable for insufficient neighbors
      coordination[i] = 0;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        dx[0] = xtmp - x[j][0];
        dx[1] = ytmp - x[j][1];
        dx[2] = ztmp - x[j][2];

        rsq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        r = sqrt(rsq);
        rinv = 1/r;

        if (rsq < cutsq) {
          //Compute Wij
          w = calc_w_quintic(i,j,dx[0],dx[1],dx[2],r);
          if (phase[j] > FixRHEO::FLUID_MAX && solid_flag)
            vj = mass[type[j]]/compute_solids->correct_rho(j,i);
          else vj = mass[type[j]]/rho[j];
          coordination[i] += 1; // Increment contributing neighbor

          M += w*vj;
        }
      }

      if (coordination[i] < N2min) continue;
      //Get inverse of 1x1 matrix
      M = 1.0/M;
      C0[i] = M;
    }

    // communicate densities - maybe not needed for ghost atoms but just to be safe
    // can remove in future once we determine they aren't necessary
    comm->forward_comm_compute(this);

  } else if (correction_order > 0) {

    int i, j, ii, jj, jnum, itype, jtype;
    int g, a, b, c, d, ib, ic, id, nperm;
    double xtmp, ytmp, ztmp, delx, dely, delz;
    double dx[3] = {0};

    //Turn off GSL error handler so we can check and revert RK to Quintic
    //when inssuficient neighbors
    gsl_set_error_handler_off();

    // Create Moment matrix M and polynomial basis vector H
    //The moment Matrix array has to be 1D to be compatible with gsl
    double H[Mdim], M[Mdim*Mdim];

    if (atom->nmax > nmax){
      nmax = atom->nmax;
      memory->destroy(C);
      memory->create(C, nmax,ncor,Mdim, "rheo/kernel:C");
    }

    // Solve linear system for each atom
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2]; // Will be ignored in 2D

      jlist = firstneigh[i];
      jnum = numneigh[i];
      itype = type[i];

      //Initialize M and H to zero:
      for (a = 0; a < Mdim; a++) {
        for (b = a; b < Mdim; b++) {
          //Just zero upper-triangle of M since it will be symmetric
          M[a*Mdim+b] = 0;
        }
      }
      //Zero moment
      //variables for switching off RK caclulation if insufficient neighbors
      sflag = 0;
      coordination[i] = 0;
      // Loop over neighbors to populate elements of L
      //m0[i] = 0.0;

      //Self contribution
      //w = calc_w_quintic(i,i,0.,0.,0.,0.);
      //m0[i] += w;
      //vi = 1/vw[i];

      vi = mass[type[i]]/rho[i]; // Overwritten if i is solid

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        dx[0] = xtmp - x[j][0];
        dx[1] = ytmp - x[j][1];
        dx[2] = ztmp - x[j][2];

        rsq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
        r = sqrt(rsq);
        rinv = 1/r;

        if (rsq < cutsq) {
          //Compute Wij
          w = calc_w_quintic(i,j,dx[0],dx[1],dx[2],r);
          //vj = 1/vw[j]; //

          if (phase[i] > FixRHEO::FLUID_MAX && solid_flag)
            vi = mass[type[i]]/compute_solids->correct_rho(i,j);

          if (phase[j] > FixRHEO::FLUID_MAX && solid_flag)
            vj = mass[type[j]]/compute_solids->correct_rho(j,i);
          else vj = mass[type[j]]/rho[j];
          //m0[i] += w;

          //Populate the H-vector of polynomials (2D)
          if (dim == 2) {
            H[0] = 1.0;
            H[1] = dx[0]*cutinv;
            H[2] = dx[1]*cutinv;
            if (kernel_type == CRK2) {
              H[3] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
              H[4] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
              H[5] = dx[0]*dx[1]*cutinv*cutinv;
            }
          } else {
            H[0] = 1.0;
            H[1] = dx[0]*cutinv;
            H[2] = dx[1]*cutinv;
            H[3] = dx[2]*cutinv;
            if (kernel_type == CRK2) {
              H[4] = 0.5*dx[0]*dx[0]*cutinv*cutinv;
              H[5] = 0.5*dx[1]*dx[1]*cutinv*cutinv;
              H[6] = 0.5*dx[2]*dx[2]*cutinv*cutinv;
              H[7] = dx[0]*dx[1]*cutinv*cutinv;
              H[8] = dx[0]*dx[2]*cutinv*cutinv;
              H[9] = dx[1]*dx[2]*cutinv*cutinv;
            }
          }
          coordination[i] += 1; // Increment contributing neighbor

          //Populate the upper triangle of the
          for (a = 0; a < Mdim; a++) {
            for (b = a; b < Mdim; b++) {
              M[a*Mdim+b] += H[a]*H[b]*w*vj;
            }
          }
        }
      }
      //Now populate the lower triangle from the symmetric entries of M:
      for (a = 0; a < Mdim; a++) {
        for (b = a; b < Mdim; b++) {
          M[b*Mdim+a] = M[a*Mdim+b];
        }
      }

      if (coordination[i] < N2min) continue;

      //Use gsl to get Minv
      //Since the polynomials are independent, M is symmetrix & positive-definite
      //So we will use a Cholesky decomposition
      gsl_matrix_view gM = gsl_matrix_view_array(M,Mdim,Mdim);
      int status;
      status = 0;
      //gsl_linalg_cholesky_decomp1 -> gsl_linalg_cholesky_decomp
      //So I don't have to update my gsl (me being lazy, can revert)
      status = gsl_linalg_cholesky_decomp(&gM.matrix);  //M is now the cholesky decomposition of M
      // check if erro in inversion
      if (status) {
        //Revert to uncorrected SPH for this particle
        gsl_error_flag = 1;
        gsl_error_tags.insert(tag[i]);

        //check if not positive-definite
        if (status != GSL_EDOM)
          fprintf(stderr, "failed, gsl_errno=%d.n", status);

        continue;
      } else {
        gsl_linalg_cholesky_invert(&gM.matrix);   //M is now M^-1
      }

      // Correction coefficients are just the columns of M^-1 multiplied by an appropriate coefficient
      //Solve the linear system several times to get coefficientns
      // M:    1   x   y  (z)  x^2  y^2 (z^2) xy   (xz)   (yz)
      //----------------------------------------------------------
      //       0   1   2       3     4        5                 || 2D indexing
      //       0   1   2   3   4     5   6    7     8      9    || 3D indexing
      //  W    1   .   .   .    .    .   .    .     .      .
      // dWx   .  -1   .   .    .    .   .    .     .      .
      // dWy   .   .  -1   .    .    .   .    .     .      .
      // dWz   .   .   .  (-1)  .    .   .    .     .      .
      // d2Wx  .   .   .   .    2    .   .    .     .      .
      // d2Wy  .   .   .   .    .    2   .    .     .      .
      // d2Wz  .   .   .   .    .    .  (2)   .     .      .

      //0 1 2 3 4
      //0 1 2 3 4 5 6

      // Use coefficents to compute smoothed density
      //Pack coefficients into C
      for (a = 0; a < Mdim; a++) {
        //W
        C[i][0][a] = M[a*Mdim + 0]; // all rows of column 0
        for (b = 0; b < dim; b++) {
          //First derivatives
          C[i][1+b][a] = -M[a*Mdim + b+1]/cut;  // columns 1-2 (2D)  or 1-3 (3D)
          //Second derivatives
          if (kernel_type == CRK2)
            C[i][1+dim+b][a] = M[a*Mdim + b+1+dim]/cutsq; // columns 3-4 (2D) or 4-6 (3D)
        }
      }
    }

    // communicate densities - maybe not needed for ghost atoms but just to be safe
    // can remove in future once we determine they aren't necessary
    comm->forward_comm_compute(this);
  }
}

/* ---------------------------------------------------------------------- */

int ComputeRHEOKernel::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,a,b;
  coordination = atom->ivector[index_coord];
  m = 0;
  if (correction_order > 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (a = 0; a < ncor; a++) {
        for (b = 0; b < Mdim; b++) {
          buf[m++] = C[j][a][b];
        }
      }
      buf[m++] = coordination[j];
    }
  } else if (kernel_type == CRK0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = C0[j];
      buf[m++] = coordination[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = coordination[j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeRHEOKernel::unpack_forward_comm(int n, int first, double *buf)
{
  int i, k, m, last,a,b;
  coordination = atom->ivector[index_coord];
  m = 0;
  last = first + n;
  if (correction_order > 0) {
    for (i = first; i < last; i++) {
      for (a = 0; a < ncor; a++) {
        for (b = 0; b < Mdim; b++) {
          C[i][a][b] = buf[m++];
        }
      }
      coordination[i] = buf[m++];
    }
  } else if (kernel_type == CRK0) {
    for (i = first; i < last; i++) {
      C0[i] = buf[m++];
      coordination[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++) {
      coordination[i] = buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeRHEOKernel::memory_usage()
{
  double bytes = 0;
  if (kernel_type == CRK2) {
      int n1 = 1 + dim + dim*dim; // A, B, G
      bytes += nmax * n1 * sizeof(double);
  }
  if (memory_flag) {
    bytes += locations_w.size()*sizeof(int);
    bytes += locations_wp.size()*sizeof(int);
    bytes += max_nstored*nstored_w*sizeof(double);
    bytes += max_nstored*nstored_wp*(1+2*dim)*sizeof(double);
  }
  return bytes;
}


