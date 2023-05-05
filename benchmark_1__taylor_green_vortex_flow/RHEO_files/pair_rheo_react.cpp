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

/* ----------------------------------------------------------------------
   Contributing authors: Leo Silbert (SNL), Gary Grest (SNL)
------------------------------------------------------------------------- */

#include "pair_rheo_react.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_dummy.h"
#include "fix_neigh_history.h"
#include "fix_rheo.h"
#include "comm.h"
#include "group.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRHEOReact::PairRHEOReact(LAMMPS *lmp) : Pair(lmp),
  dbond(NULL)
{
  single_enable = 0;
  size_history = 2;
  beyond_contact = 1; //May not need, check flag first though
  comm_reverse = 1;
  comm_forward = 1;

  // create dummy fix as placeholder for FixNeighHistory
  // this is so final order of Modify:fix will conform to input script
  // can't declare fix history here b/c some commands (e.g. displace/atom)
  // will call irregular communication before the fix is initialized
  fix_history = nullptr;
  modify->add_fix("NEIGH_HISTORY_RHEO_REACT_DUMMY all DUMMY");
  fix_dummy = (FixDummy *) modify->fix[modify->nfix-1];

  //Store persistent per atom quantities
  char **fixarg = new char*[6];
  fixarg[0] = (char *) "PROPERTY_ATOM_RHEO_REACT";
  fixarg[1] = (char *) "all";
  fixarg[2] = (char *) "property/atom";
  fixarg[3] = (char *) "i_rheo_react_nbond";
  fixarg[4] = (char *) "ghost";
  fixarg[5] = (char *) "yes";
  modify->add_fix(6,fixarg,1);

  int temp_flag;
  index_nb = atom->find_custom("rheo_react_nbond", temp_flag);
  if ((index_nb < 0) || (temp_flag != 0))
      error->all(FLERR, "Pair rheo/react can't find fix property/atom bond number");

  delete [] fixarg;

  //Store non-persistent per atom quantities, intermediate
  nmax  = atom->nmax;
  memory->create(dbond, nmax, "rheo/react:dbond");
}

/* ---------------------------------------------------------------------- */

PairRHEOReact::~PairRHEOReact()
{
  if (modify->nfix && fix_history) modify->delete_fix("NEIGH_HISTORY_RHEO_REACT");
  if (modify->nfix && fix_dummy) modify->delete_fix("NEIGH_HISTORY_RHEO_REACT_DUMMY");
  if (modify->nfix) modify->delete_fix("PROPERTY_ATOM_RHEO_REACT");

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutbsq);

    memory->destroy(cut);
    memory->destroy(cutbond);
    memory->destroy(k);
    memory->destroy(eps);
    memory->destroy(gamma);
    memory->destroy(t_form);
    memory->destroy(rlimit);
    memory->destroy(sigma);
    memory->destroy(krepel);
  }

  memory->destroy(dbond);
}


/* ---------------------------------------------------------------------- */

void PairRHEOReact::add_bonds(int groupbit)
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r;
  int itype, jtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *bond,**firstbond;
  double *data,*alldata,**firstdata;

  int *nbond = atom->ivector[index_nb];
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *phase = atom->phase;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firstbond = fix_history->firstflag;
  firstdata = fix_history->firstvalue;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    bond = firstbond[i];
    alldata = firstdata[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    if(phase[i] != FixRHEO::REACTIVE) continue;
    if(! mask[i] & groupbit) continue;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      if(phase[j] != FixRHEO::REACTIVE) continue;
      if(! mask[j] & groupbit) continue;

      // Check if not bonded
      if(!bond[jj]) {
        // Bond if in range
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutbsq[itype][jtype]) {
          r = sqrt(rsq);
          bond[jj] = 1;
          data = &alldata[2*jj];
          data[0] = r;
          data[1] = 0;
          dbond[i] ++;
          dbond[j] ++;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairRHEOReact::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r,rinv,r0;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double fpair,dot,evdwl,smooth;
  int itype, jtype;

  int *ilist,*jlist,*numneigh,**firstneigh;
  int *saved,**firstsaved;
  double *data,*alldata,**firstdata;

  ev_init(eflag,vflag);

  int bondupdate = 1;
  if (update->setupflag) bondupdate = 0;

  dt = update->dt;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *phase = atom->phase;
  int *mask = atom->mask;
  int *nbond = atom->ivector[index_nb];
  int *surface = atom->surface;
  double *rsurf = atom->dvector[index_rsurf];

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firstsaved = fix_history->firstflag;
  firstdata = fix_history->firstvalue;

  if (atom->nmax > nmax){
    nmax = atom->nmax;
    memory->destroy(dbond);
    memory->create(dbond, nmax, "rheo/react:dbond");
  }

  // Switch to no shift if it has bonds (could have just been changed from reactive)
  for(i = 0; i < nmax; i++) {
    dbond[i] = 0;
  }

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    saved = firstsaved[i];
    alldata = firstdata[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      data = &alldata[2*jj];

      // If not bonded and there's an internal fluid particle, unsave any data and skip
      if (!(saved[jj] == 1 && data[0] > 0)) {
        if ((phase[i] <= FixRHEO::FLUID_MAX && rsurf[i] > rlimit[itype][jtype]) || (phase[j] <= FixRHEO::FLUID_MAX && rsurf[j] > rlimit[itype][jtype])) {
          saved[jj] = 0;
          continue;
        }
      }

      // If both are solid, unbond and skip
      if ((phase[i] == FixRHEO::SOLID || phase[i] == FixRHEO::FREEZING) &&
          (phase[j] == FixRHEO::SOLID || phase[j] == FixRHEO::FREEZING)) {
        //If bonded, deincrement
        if (saved[jj] == 1 && data[0] > 0) {
          dbond[i] --;
          dbond[j] --;
        }
        saved[jj] = 0;
        continue;
      }

      // Remaining options are react+sold, react+react, react+surf/fluid, or surf/fluid+surf/fluid

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      // If unbonded and beyond bond distance, unsave and skip
      if (data[0] == -1 && rsq > cutbsq[itype][jtype]) {
        saved[jj] = 0;
        continue;
      }

      r = sqrt(rsq);
      // Initialize data if not currently saved since all could bond if they are on the surface
      if (saved[jj] == 0) {
        data[0] = -1;
        data[1] = 0;
        saved[jj] = 1;
      }

      // Check for bond formation (unbonded) or breakage (bonded)
      if (data[0] == -1) {
        // If unbonded, check if we count down to bonding if both on surface (not given for r or s)
        if (bondupdate && rsurf[i] <= rlimit[itype][jtype] && rsurf[j] <= rlimit[itype][jtype]) {
          data[1] += dt;
          if (data[1] >= t_form[itype][jtype]) {
            data[0] = r;
            dbond[i] ++;
            dbond[j] ++;
            data[1] = 0;
          }
        }
      } else {
        // If bonded, check if breaks in tension
        r0 = data[0];
        if (r > ((1.0+eps[itype][jtype])*r0)) {
          saved[jj] = 0;
          dbond[i] --;
          dbond[j] --;
          data[0] = -1;
        }
      }

      // Apply forces
      delvx = vxtmp - v[j][0];
      delvy = vytmp - v[j][1];
      delvz = vztmp - v[j][2];
      rinv = 1.0/r;

      if (data[0] <= 0) {
        // Skip if either is fluid (only include r+s or r+r since already skipped s+s)
        if (phase[i] <= FixRHEO::FLUID_MAX || phase[j] <= FixRHEO::FLUID_MAX) continue;

        // Skip if out of contact
        if (rsq > sigma[itype][jtype]*sigma[itype][jtype]) continue;

        fpair = krepel[itype][jtype]*(sigma[itype][jtype]-r);
        if (eflag)
          evdwl = -0.5*krepel[itype][jtype]*(sigma[itype][jtype]-r)*(sigma[itype][jtype]-r);

        smooth = rsq/(sigma[itype][jtype]*sigma[itype][jtype]);
        smooth *= smooth;
        smooth = 1.0 - smooth;
        dot = delx*delvx + dely*delvy + delz*delvz;
        fpair -= gamma[itype][jtype]*dot*smooth*rinv;

        fpair *= rinv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
      } else {
        // Bonded
        r0 = data[0];

        fpair = k[itype][jtype]*(r0-r);
        if (evflag) evdwl = -0.5*fpair*(r0-r);

        dot = delx*delvx + dely*delvy + delz*delvz;
        fpair -= gamma[itype][jtype]*dot*rinv;

        smooth = 1.0;
        if (r > r0) {
          smooth = (r-r0)/(r0*eps[itype][jtype]);
          smooth *= smooth;
          smooth *= smooth;
          smooth = 1 - smooth;
        }

        fpair *= rinv*smooth;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  // Communicate changes in nbond
  if(newton_pair) comm->reverse_comm_pair(this);

  for(i = 0; i < nlocal; i++) {
    nbond[i] += dbond[i];

    // If it has bonds it is reactive (no shifting)
    // If a reactive particle breaks all bonds, return to fluid
    // Keep it non-shifting for this timestep to be safe
    if (nbond[i] != 0 && phase[i] <= FixRHEO::FLUID_MAX) phase[i] = FixRHEO::FLUID_NO_SHIFT;
   // if (nbond[i] == 0 && phase[i] == FixRHEO::REACTIVE) phase[i] = FixRHEO::FLUID_NO_SHIFT;

  }

  // Update new phases in case other fixes/computes need ghost phase
  comm->forward_comm_pair(this);
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairRHEOReact::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(cutbond,n+1,n+1,"pair:cutbond");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cutbsq,n+1,n+1,"pair:cutbsq");
  memory->create(k,n+1,n+1,"pair:k");
  memory->create(eps,n+1,n+1,"pair:eps");
  memory->create(gamma,n+1,n+1,"pair:gamma");
  memory->create(t_form,n+1,n+1,"pair:t_form");
  memory->create(rlimit,n+1,n+1,"pair:rlimit");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(krepel,n+1,n+1,"pair:krepel");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRHEOReact::settings(int narg, char **arg)
{
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairRHEOReact::coeff(int narg, char **arg)
{
  if (narg != 11) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1, atom->ntypes, ilo, ihi,error);
  utils::bounds(FLERR,arg[1],1, atom->ntypes, jlo, jhi,error);

  double cut_one = utils::numeric(FLERR,arg[2],false,lmp);
  double cutb_one = utils::numeric(FLERR,arg[3],false,lmp);
  double k_one = utils::numeric(FLERR,arg[4],false,lmp);
  double eps_one = utils::numeric(FLERR,arg[5],false,lmp);
  double gamma_one = utils::numeric(FLERR,arg[6],false,lmp);
  double t_form_one = utils::numeric(FLERR,arg[7],false,lmp);
  double rlimit_one = utils::numeric(FLERR,arg[8],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[9],false,lmp);
  double krepel_one = utils::numeric(FLERR,arg[10],false,lmp);

  if (k_one < 0.0 || eps_one < 0.0 ||
   t_form_one < 0.0 || (1.0+eps_one)*cutb_one > cut_one)
     error->all(FLERR,"Illegal pair_style command");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      cutbond[i][j] = cutb_one;
      k[i][j] = k_one;
      eps[i][j] = eps_one;
      gamma[i][j] = gamma_one;
      t_form[i][j] = t_form_one;
      rlimit[i][j] = rlimit_one;
      sigma[i][j] = sigma_one;
      krepel[i][j] = krepel_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
  init specific to this pair style
------------------------------------------------------------------------- */

void PairRHEOReact::init_style()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->history = 1;

  if (fix_history == nullptr) {

    // Don't want history[i][j] = -history[j][i]
    nondefault_history_transfer = 1;

    char dnumstr[16];
    sprintf(dnumstr,"%d",size_history);
    char **fixarg = new char*[4];
    fixarg[0] = (char *) "NEIGH_HISTORY_RHEO_REACT";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "NEIGH_HISTORY";
    fixarg[3] = dnumstr;
    modify->replace_fix("NEIGH_HISTORY_RHEO_REACT_DUMMY",4,fixarg,1);
    delete [] fixarg;
    int ifix = modify->find_fix("NEIGH_HISTORY_RHEO_REACT");
    fix_history = (FixNeighHistory *) modify->fix[ifix];
    fix_history->pair = this;
    fix_history->use_bit_flag = 0;
    fix_dummy = nullptr;
  }

  int temp_flag;
  index_rsurf = atom->find_custom("rsurf", temp_flag);
  if ((index_rsurf < 0) || (temp_flag != 1))
      error->all(FLERR, "Pair rheo/react can't find fix property/atom rsurf");
}

/* ----------------------------------------------------------------------
 setup specific to this pair style
 ------------------------------------------------------------------------- */

void PairRHEOReact::setup() {
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Using pair rheo/react without fix rheo");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);

  ifix = modify->find_fix_by_style("rheo/surface");
  if (ifix == -1) error->all(FLERR, "Using pair rheo/react without fix rheo/surface");

  if (force->newton_pair == 0) error->all(FLERR,
      "Pair rheo/react needs newton pair on for bond changes to be consistent");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairRHEOReact::init_one(int i, int j)
{
  if (setflag[i][j] == 0)  error->all(FLERR,"All pair coeffs are not set");

  double einv = 1/eps[i][j];

  cutbsq[i][j] = cutbond[i][j]*cutbond[i][j];

  cutbsq[j][i] = cutbsq[i][j];
  cut[j][i] = cut[i][j];
  cutbond[j][i] = cutbond[i][j];
  k[j][i] = k[i][j];
  eps[j][i] = eps[i][j];
  gamma[j][i] = gamma[i][j];
  t_form[j][i] = t_form[i][j];
  rlimit[j][i] = rlimit[i][j];
  sigma[j][i] = sigma[i][j];
  krepel[j][i] = krepel[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairRHEOReact::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++)
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j],sizeof(double),1,fp);
        fwrite(&cutbond[i][j],sizeof(double),1,fp);
        fwrite(&k[i][j],sizeof(double),1,fp);
        fwrite(&eps[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&t_form[i][j],sizeof(double),1,fp);
        fwrite(&rlimit[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&krepel[i][j],sizeof(double),1,fp);
      }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairRHEOReact::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&cut[i][j],sizeof(double),1,fp);
          fread(&cutbond[i][j],sizeof(double),1,fp);
          fread(&k[i][j],sizeof(double),1,fp);
          fread(&eps[i][j],sizeof(double),1,fp);
          fread(&gamma[i][j],sizeof(double),1,fp);
          fread(&t_form[i][j],sizeof(double),1,fp);
          fread(&rlimit[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&krepel[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cutbond[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&k[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&eps[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&t_form[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rlimit[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&krepel[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}




/* ----------------------------------------------------------------------
   transfer history during fix/neigh/history exchange - transfer same sign
------------------------------------------------------------------------- */

void PairRHEOReact::transfer_history(double* source, double* target)
{
  for (int i = 0; i < size_history; i++)
    target[i] = source[i];
}

/* ---------------------------------------------------------------------- */

int PairRHEOReact::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    buf[m++] = dbond[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairRHEOReact::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    dbond[j] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairRHEOReact::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;
  int *phase = atom->phase;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = phase[j];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void PairRHEOReact::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;
  int *phase = atom->phase;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    phase[i] = buf[m++];
  }
}