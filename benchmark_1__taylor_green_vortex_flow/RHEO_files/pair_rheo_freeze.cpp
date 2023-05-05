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

#include "pair_rheo_freeze.h"
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
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRHEOFreeze::PairRHEOFreeze(LAMMPS *lmp) : Pair(lmp),
  dbond(NULL)
{
  single_enable = 0;
  size_history = 1;
  beyond_contact = 1; //May not need, check flag first though
  comm_reverse = 1;
  expand_flag = 0;

  // create dummy fix as placeholder for FixNeighHistory
  // this is so final order of Modify:fix will conform to input script
  // can't declare fix history here b/c some commands (e.g. displace/atom)
  // will call irregular communication before the fix is initialized
  fix_history = nullptr;
  modify->add_fix("NEIGH_HISTORY_RHEO_FREEZE_DUMMY all DUMMY");
  fix_dummy = (FixDummy *) modify->fix[modify->nfix-1];

  //Store persistent per atom quantities
  char **fixarg = new char*[4];
  fixarg[0] = (char *) "PROPERTY_ATOM_RHEO_FREEZE";
  fixarg[1] = (char *) "all";
  fixarg[2] = (char *) "property/atom";
  fixarg[3] = (char *) "i_rheo_freeze_nbond";
  modify->add_fix(4,fixarg,1);

  int temp_flag;
  index_nb = atom->find_custom("rheo_freeze_nbond", temp_flag);
  if ((index_nb < 0) || (temp_flag != 0))
      error->all(FLERR, "Pair rheo/freeze can't find fix property/atom bond number");

  delete [] fixarg;

  //Store non-persistent per atom quantities, intermediate
  nmax  = atom->nmax;
  memory->create(dbond, nmax, "rheo/freeze:dbond");
}

/* ---------------------------------------------------------------------- */

PairRHEOFreeze::~PairRHEOFreeze()
{
  if (modify->nfix && fix_history) modify->delete_fix("NEIGH_HISTORY_RHEO_FREEZE");
  if (modify->nfix && fix_dummy) modify->delete_fix("NEIGH_HISTORY_RHEO_FREEZE_DUMMY");
  if (modify->nfix) modify->delete_fix("PROPERTY_ATOM_RHEO_FREEZE");

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutbsq);

    memory->destroy(cut);
    memory->destroy(cutbond);
    memory->destroy(k);
    memory->destroy(sigma);
    memory->destroy(eps);
    memory->destroy(gamma);

    memory->destroy(tscale);
    memory->destroy(tcrit);
    memory->destroy(texpand);
  }

  memory->destroy(dbond);
}

/* ---------------------------------------------------------------------- */

void PairRHEOFreeze::add_bonds(int groupbit)
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r;
  int itype, jtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *bond,**firstbond;
  double *allr0,**firstr0;

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
  firstr0 = fix_history->firstvalue;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    bond = firstbond[i];
    allr0 = firstr0[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    if(phase[i] != FixRHEO::FREEZING) continue;
    if(! mask[i] & groupbit) continue;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      if(phase[j] != FixRHEO::FREEZING) continue;
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
          allr0[jj] = r;
          dbond[i] ++;
          dbond[j] ++;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairRHEOFreeze::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r,rinv,r0;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz,smooth;
  double fpair,dot,evdwl,temp_ave,dtemp;
;
  int itype, jtype;

  int *ilist,*jlist,*numneigh,**firstneigh;
  int *bond,**firstbond;
  double *allr0,**firstr0;

  ev_init(eflag,vflag);

  int bondupdate = 1;
  if (update->setupflag) bondupdate = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *temp = atom->temp;
  int *phase = atom->phase;
  int *type = atom->type;
  int *mask = atom->mask;

  int *nbond = atom->ivector[index_nb];

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  firstbond = fix_history->firstflag;
  firstr0 = fix_history->firstvalue;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(dbond);
    memory->create(dbond, nmax, "rheo/freeze:dbond");
  }

  for(i = 0; i < nmax; i++)
    dbond[i] = 0;

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
    bond = firstbond[i];
    allr0 = firstr0[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      // If either is non-solid, unbond and skip
      if((phase[i] != FixRHEO::SOLID and phase[i] != FixRHEO::FREEZING) or
          (phase[j] != FixRHEO::SOLID and phase[j] != FixRHEO::FREEZING)){
        //If bonded, deincrement
        if(bond[jj] == 1){
          bond[jj] = 0;
          dbond[i] --;
          dbond[j] --;
        }
        continue;
      }

     // if(atom->tag[i] == 27139 or atom->tag[j] == 27139) printf("Solid forces\n");
      // First process changes in bonds before applying forces
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delvx = vxtmp - v[j][0];
      delvy = vytmp - v[j][1];
      delvz = vztmp - v[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      rinv = 1.0/r;
      jtype = type[j];

      // If freezing, check if can bond
      if (phase[i] == FixRHEO::FREEZING || phase[j] == FixRHEO::FREEZING) {
        if (rsq < cutbsq[itype][jtype]) {
          r = sqrt(rsq);
          bond[jj] = 1;
          allr0[jj] = r;
          dbond[i] ++;
          dbond[j] ++;
        }
      }

      // If bonded, check if breaks in tension
      if (bond[jj]) {
        r = sqrt(rsq);
        r0 = allr0[jj];
        if (r > (1.0+eps[itype][jtype])*r0) {
          bond[jj] = 0;
          dbond[i] --;
          dbond[j] --;
        }
      }

      // Apply forces
      if(!bond[jj]) {
        // Not bonded

        // Skip if out of contact
        if(rsq > sigma[itype][jtype]*sigma[itype][jtype]) continue;

        fpair = k[itype][jtype]*(sigma[itype][jtype]-r);
        if (eflag)
          evdwl = -0.5*k[itype][jtype]*(sigma[itype][jtype]-r)*(sigma[itype][jtype]-r);

        smooth = rsq/(sigma[itype][jtype]*sigma[itype][jtype]);
        smooth *= smooth;
        smooth = 1.0 - smooth;
        dot = delx*delvx + dely*delvy + delz*delvz;
        fpair -= gamma[itype][jtype]*dot*smooth*rinv;

        fpair *= rinv;

        // Add forces
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
        // Bonded, r, r0 already defined

        if (expand_flag) {
          temp_ave = temp[i];
          if (temp_ave < temp[j]) temp_ave = temp[j];
          dtemp = tcrit[itype][jtype] - temp_ave;
          if (dtemp > 0.0) {
            smooth = dtemp/tscale[itype][jtype];
            if (smooth > 1.0) smooth = 1.0;
            r0 *= 1.0 + texpand[itype][jtype]*smooth;
          }
        }

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

        // Add forces
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
    if (phase[i] == FixRHEO::FREEZING)
      phase[i] = FixRHEO::SOLID;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairRHEOFreeze::allocate()
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
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(eps,n+1,n+1,"pair:eps");
  memory->create(gamma,n+1,n+1,"pair:gamma");

  memory->create(tscale,n+1,n+1,"pair:tscale");
  memory->create(tcrit,n+1,n+1,"pair:tcrit");
  memory->create(texpand,n+1,n+1,"pair:texpand");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRHEOFreeze::settings(int narg, char **arg)
{
  if (narg > 1) error->all(FLERR,"Illegal pair_style command");

  if (narg == 1) {
    if (strcmp(arg[0], "thermal/expand") == 0) {
      expand_flag = 1;
    } else {
      error->all(FLERR, "Illegal pair_style command");
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairRHEOFreeze::coeff(int narg, char **arg)
{
  if (narg < 8) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1, atom->ntypes, ilo, ihi,error);
  utils::bounds(FLERR,arg[1],1, atom->ntypes, jlo, jhi,error);


  double cut_one = utils::numeric(FLERR,arg[2],false,lmp);
  double cutb_one = utils::numeric(FLERR,arg[3],false,lmp);
  double k_one = utils::numeric(FLERR,arg[4],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[5],false,lmp);
  double eps_one = utils::numeric(FLERR,arg[6],false,lmp);
  double gamma_one = utils::numeric(FLERR,arg[7],false,lmp);

  if (k_one < 0.0 || eps_one < 0.0 ||
   (1.0+eps_one)*cutb_one > cut_one)
    error->all(FLERR,"Illegal pair_style command");

  double texpand_one, tcrit_one, tscale_one;
  if (expand_flag) {
    if (narg < 12) error->all(FLERR, "Illegal pair_style command");
    texpand_one = utils::numeric(FLERR,arg[9],false,lmp);
    tcrit_one = utils::numeric(FLERR,arg[10],false,lmp);
    tscale_one = utils::numeric(FLERR,arg[11],false,lmp);
  } else if (narg > 8) {
    error->all(FLERR, "Illegal pair_style command");
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      cutbond[i][j] = cutb_one;
      k[i][j] = k_one;
      sigma[i][j] = sigma_one;
      eps[i][j] = eps_one;
      gamma[i][j] = gamma_one;
      setflag[i][j] = 1;

      if (expand_flag) {
        texpand[i][j] = texpand_one;
        tcrit[i][j] = tcrit_one;
        tscale[i][j] = tscale_one;
      }

      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
  init specific to this pair style
------------------------------------------------------------------------- */

void PairRHEOFreeze::init_style()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->history = 1;

  if (fix_history == nullptr) {

    // Don't want history[i][j] = -history[j][i]
    nondefault_history_transfer = 1;

    char dnumstr[16];
    sprintf(dnumstr,"%d",size_history);
    char **fixarg = new char*[4];
    fixarg[0] = (char *) "NEIGH_HISTORY_RHEO_FREEZE";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "NEIGH_HISTORY";
    fixarg[3] = dnumstr;
    modify->replace_fix("NEIGH_HISTORY_RHEO_FREEZE_DUMMY",4,fixarg,1);
    delete [] fixarg;
    int ifix = modify->find_fix("NEIGH_HISTORY_RHEO_FREEZE");
    fix_history = (FixNeighHistory *) modify->fix[ifix];
    fix_history->pair = this;
    fix_history->use_bit_flag = 0;
    fix_dummy = nullptr;
  }
}


/* ----------------------------------------------------------------------
   setup specific to this pair style
------------------------------------------------------------------------- */

void PairRHEOFreeze::setup()
{
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Using pair rheo/freeze without fix rheo");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);

  if (force->newton_pair == 0) error->all(FLERR,
      "Pair rheo/freeze needs newton pair on for bond changes to be consistent");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairRHEOFreeze::init_one(int i, int j)
{
  if (setflag[i][j] == 0)  error->all(FLERR,"All pair coeffs are not set");

  cutbsq[i][j] = cutbond[i][j]*cutbond[i][j];

  cutbsq[j][i] = cutbsq[i][j];
  cut[j][i] = cut[i][j];
  cutbond[j][i] = cutbond[i][j];
  k[j][i] = k[i][j];
  eps[j][i] = eps[i][j];
  sigma[j][i] = sigma[i][j];
  gamma[j][i] = gamma[i][j];

  if (expand_flag) {
    texpand[j][i] = texpand[i][j];
    tcrit[j][i] = tcrit[i][j];
    tscale[j][i] = tscale[i][j];
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairRHEOFreeze::write_restart(FILE *fp)
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
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&texpand[i][j],sizeof(double),1,fp);
        fwrite(&tcrit[i][j],sizeof(double),1,fp);
        fwrite(&tscale[i][j],sizeof(double),1,fp);
      }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairRHEOFreeze::read_restart(FILE *fp)
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
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&gamma[i][j],sizeof(double),1,fp);
          fread(&texpand[i][j],sizeof(double),1,fp);
          fread(&tcrit[i][j],sizeof(double),1,fp);
          fread(&tscale[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cutbond[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&k[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&eps[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&texpand[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&tcrit[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&tscale[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}


/* ----------------------------------------------------------------------
   transfer history during fix/neigh/history exchange - transfer same sign
------------------------------------------------------------------------- */

void PairRHEOFreeze::transfer_history(double* source, double* target)
{
  for (int i = 0; i < size_history; i++)
    target[i] = source[i];
}

/* ---------------------------------------------------------------------- */

int PairRHEOFreeze::pack_reverse_comm(int n, int first, double *buf)
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

void PairRHEOFreeze::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    dbond[j] += buf[m++];
  }
}


