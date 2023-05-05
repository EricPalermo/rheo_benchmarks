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

#include "pair_rheo.h"
#include "compute_rheo_kernel.h"
#include "compute_rheo_grad.h"
#include "compute_rheo_solids.h"
#include "fix_rheo.h"
#include "fix_store.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "modify.h"
#include "error.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;
enum {LINEAR, CUBIC, TAITWATER};

/* ---------------------------------------------------------------------- */

PairRHEO::PairRHEO(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  compute_kernel = NULL;
  compute_grad = NULL;
  compute_solids = NULL;
  fix_rheo = NULL;
  single_enable = 0;

  pressure_style = LINEAR;

  artificial_visc_flag = 0;
  rho_damp_flag = 0;
  thermal_flag = 0;

  laplacian_order = -1;
}

/* ---------------------------------------------------------------------- */

PairRHEO::~PairRHEO()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(rho0);
    memory->destroy(csq);
  }
}

/* ---------------------------------------------------------------------- */

void PairRHEO::compute(int eflag, int vflag)
{
  int i, j, a, b, ii, jj, inum, jnum, itype, jtype;
  int error_flag, pair_force_flag, pair_rho_flag, pair_avisc_flag;
  double xtmp, ytmp, ztmp;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double imass, jmass, rsq, r, ir;

  double w, wp, rhoi, rhoj, voli, volj, Pi, Pj;
  double *dWij, *dWji, *d2Wij, *d2Wji, *dW1ij, *dW1ji;
  double vijeij, etai, etaj, kappai, kappaj;
  double Ti, Tj, dT;
  double drho_damp, fmag;
  double invhthird, mu, q, cs, fp_prefactor;
  double dx[3] = {0};
  double fv[3] = {0};
  double dfp[3] = {0};
  double fsolid[3] = {0};
  double du[3] = {0};
  double vi[3] = {0};
  double vj[3] = {0};
  double dv[3] = {0};
  double psi_ij = 0.0;
  double Fij = 0.0;

  ev_init(eflag, vflag);

  double **gradv = compute_grad->gradv;
  double **gradt = compute_grad->gradt;
  double **gradr = compute_grad->gradr;
  double **gradn = compute_grad->gradn;
  double **v = atom->v;
  double **x = atom->x;
  double **f = atom->f;
  double **fp = atom->fp;
  double *rho = atom->rho;
  double *mass = atom->mass;
  double *drho = atom->drho;
  double *temp = atom->temp;
  double *heat = atom->heat;
  double *viscosity = atom->viscosity;
  double *conductivity = atom->conductivity;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *phase = atom->phase;

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int dim = domain->dimension;

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
    imass = mass[itype];

    kappai = conductivity[i];
    etai = viscosity[i];
    Ti = temp[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      dx[0] = xtmp - x[j][0];
      dx[1] = ytmp - x[j][1];
      dx[2] = ztmp - x[j][2];
      rsq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
      jtype = type[j];
      jmass = mass[jtype];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        ir = 1/r;

        rhoi = rho[i];
        rhoj = rho[j];
        kappaj = conductivity[j];
        etaj = viscosity[j];
        Tj = temp[j];

        pair_rho_flag = 0;
        pair_force_flag = 0;
        pair_avisc_flag = 0;
        if (phase[i] <= FixRHEO::FLUID_MAX || phase[j] <= FixRHEO::FLUID_MAX) {
          pair_force_flag = 1;
        }
        if (phase[i] <= FixRHEO::FLUID_MAX && phase[j] <= FixRHEO::FLUID_MAX) {
          pair_avisc_flag = 1;
          pair_rho_flag = 1;
        }

        wp = compute_kernel->calc_dw(i, j, dx[0], dx[1], dx[2], r);
        dWij = compute_kernel->dWij;
        dWji = compute_kernel->dWji;

        for (a = 0; a < dim; a ++) {
          vi[a] = v[i][a];
          vj[a] = v[j][a];
        }

        // Add corrections for walls
        fsolid[0] = 0.0;
        fsolid[1] = 0.0;
        fsolid[2] = 0.0;
        if (phase[i] <= FixRHEO::FLUID_MAX && phase[j] > FixRHEO::FLUID_MAX) {
          compute_solids->correct_v(v[i], v[j], vi, i, j);
          rhoj = compute_solids->correct_rho(j,i);
          // Repel if close to inner solid particle
          if (compute_solids->chi[j] > 0.9 && r < cut[itype][jtype]*0.5) {
            fmag = rho0[itype]*csq[itype]*cut[itype][jtype]*(compute_solids->chi[j] - 0.9)*(cut[itype][jtype]*0.5-r)*ir;
            fsolid[0] = fmag*dx[0];
            fsolid[1] = fmag*dx[1];
            fsolid[2] = fmag*dx[2];
          }
        } else if (phase[i] > FixRHEO::FLUID_MAX && phase[j] <= FixRHEO::FLUID_MAX) {
          compute_solids->correct_v(v[j], v[i], vj, j, i);
          rhoi = compute_solids->correct_rho(i,j);
          // Repel if close to inner solid particle
          if (compute_solids->chi[i] > 0.9 && r < cut[itype][jtype]*0.5) {
            fmag = rho0[jtype]*csq[itype]*cut[itype][jtype]*(compute_solids->chi[i] - 0.9)*(cut[itype][jtype]*0.5-r)*ir;
            fsolid[0] = fmag*dx[0];
            fsolid[1] = fmag*dx[1];
            fsolid[2] = fmag*dx[2];
          }
        } else if (phase[i] > FixRHEO::FLUID_MAX && phase[j] > FixRHEO::FLUID_MAX) {
          rhoi = 1.0;
          rhoj = 1.0;
        }

        // Compute volume and pressure after reconstructing
        voli = imass/rhoi;
        volj = jmass/rhoj;
        Pj = calc_pressure(rhoj,jtype);
        Pi = calc_pressure(rhoi,itype);

        //Check if Second order kernels will be used for eta*Lap(v)
        error_flag = 0;
        if (laplacian_order == 2) {
          error_flag = compute_kernel->calc_d2w(i, j, dx[0], dx[1], dx[2], r);
          d2Wij = compute_kernel->d2Wij;
          d2Wji = compute_kernel->d2Wji;
        }

        //Thermal Evolution
        if (thermal_flag) {
          dT = 0.0;
          for (a = 0; a < dim; a++) {
            //dT += kappaj*dWij[a]*gradt[j][a];
            //dT -= kappai*dWij[a]*gradt[i][a];
            dT += 1/1*(kappai+kappaj)*(Ti-Tj)*dx[a]*dWij[a]*ir*ir; //Assumes heat capacity and density = 1, needs to be generalized
          }
          dT *= voli*volj;
          heat[i] += dT;
        }

        // If either particle is fluid, compute hydrostatic and viscous forces
        // Compute eta*Lap(v) -  different forms depending on order of RK correction
        if (pair_force_flag) {
          //Hydrostatic pressure forces
          fp_prefactor = voli*volj*(Pj+ Pi);

          //Add artificial viscous pressure if required
          if (artificial_visc_flag && pair_avisc_flag){
            //Interpolate velocities to midpoint and use this difference for artificial viscosity
            for (a = 0; a < dim; a++) {
              du[a] = vi[a] - vj[a];
              for (b = 0; b < dim; b++) {
                du[a] -= 0.5*(gradv[i][a*dim + b] + gradv[j][a*dim + b])*dx[b];
              }
            }
            invhthird = 3.0/h;
            mu = (du[0]*dx[0] + du[1]*dx[1]+ du[2]*dx[2])*invhthird;
            mu = mu/(rsq*invhthird*invhthird + 1e-2);
            mu= MIN(0.0,mu);
            cs = 0.5*(sqrt(csq[itype])+sqrt(csq[jtype]));
            // "kinematic viscous pressure"  q = Q/rho
            q = av*(-2.0*cs*mu + 1.0*mu*mu);
            fp_prefactor += voli*volj*q*(rhoj + rhoi);
          }

          // -Grad[P + Q]
          dfp[0] = - fp_prefactor*dWij[0];
          dfp[1] = - fp_prefactor*dWij[1];
          dfp[2] = - fp_prefactor*dWij[2];

          // Now compute viscous eta*Lap[v] terms
          for (a = 0; a < dim; a ++) {
            fv[a] = 0.0;
            for (b = 0; b < dim; b++) {
              fv[a] += (etai+etaj)*(vi[a]-vj[a])*dx[b]*dWij[b]*ir*ir;
            }              
            fv[a] *= voli*volj;
          }

        } else {
          for (a = 0; a < dim; a ++) {
            fv[a] = 0;
            dfp[a] = 0;
          }
        }

        if (pair_force_flag) {
          f[i][0] += fv[0] + dfp[0] + fsolid[0];
          f[i][1] += fv[1] + dfp[1] + fsolid[1];
          f[i][2] += fv[2] + dfp[2] + fsolid[2];
          fp[i][0] += dfp[0];
          fp[i][1] += dfp[1];
          fp[i][2] += dfp[2];
        }

        // Density damping
        // conventional for low-order h
        // interpolated for RK 1 & 2  (Antuono et al, Computers & Fluids 2021)
        if (rho_damp_flag && pair_rho_flag) {
          if (laplacian_order>=1 && error_flag == 0){
            psi_ij = rhoj-rhoi;
            Fij = 0.0;
            for (a = 0; a < dim; a++){
              psi_ij += 0.5*(gradr[i][a]+gradr[j][a])*dx[a];
              Fij -= dx[a]*dWij[a];
            }
            Fij *= ir*ir;
            drho[i] += 2*rho_damp*psi_ij*Fij*volj;
          }
          else {
            drho_damp = 2*rho_damp*(rhoj-rhoi)*ir*wp;
            drho[i] -= drho_damp*volj;
          }
        }

        if (evflag) // Doesn't account for unbalanced forces
          ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, fv[0]+dfp[0], fv[1]+dfp[1], fv[2]+dfp[2], dx[0], dx[1], dx[2]);

        // Newton neighbors
        if (newton_pair || j < nlocal) {

          if (thermal_flag) {
            dT = 0.0;
            for(a = 0; a < dim; a++){
              //dT += kappai*dWji[a]*gradt[i][a];
              //dT -= kappaj*dWji[a]*gradt[j][a];
              dT += 1/1*(kappai+kappaj)*(Ti-Tj)*dx[a]*dWji[a]*ir*ir; //Assumes heat capacity and density = 1, needs to be generalized
            }
            dT *= -voli*volj;
            heat[j] -= dT;
          }

          for (a = 0; a < dim; a ++) {
            fv[a] = 0.0;
            for (b = 0; b < dim; b++) {
              //fv[a] += etai*dWji[b]*(gradv[i][a*dim+b]+gradv[i][b*dim+a]);
              //fv[a] -= etaj*dWji[b]*(gradv[j][a*dim+b]+gradv[j][b*dim+a]);
              fv[a] += (etai+etaj)*(vi[a]-vj[a])*dx[b]*dWji[b]*ir*ir;
            }
            fv[a] *= -voli*volj; // flip sign here b/c -= at accummulator
          }

        

          if (pair_force_flag) {
            for (a = 0; a < dim; a++)
              dfp[a] = fp_prefactor*dWji[a];
          }

          if (rho_damp_flag && pair_rho_flag){
            if (laplacian_order>=1 && error_flag == 0){
              Fij = 0.0;
              for (a = 0; a < dim; a++){
                Fij += dx[a]*dWji[a];
              }
              Fij *= ir*ir;
              psi_ij *= -1;
              drho[j] += 2*rho_damp*psi_ij*Fij*voli;
            }
            else {
              drho_damp = 2*rho_damp*(rhoj-rhoi)*ir*wp;
              drho[j] += drho_damp*voli;
            }
          }
          if (pair_force_flag) {
            f[j][0] -= fv[0] + dfp[0] + fsolid[0];
            f[j][1] -= fv[1] + dfp[1] + fsolid[1];
            f[j][2] -= fv[2] + dfp[2] + fsolid[2];

            fp[j][0] -= dfp[0];
            fp[j][1] -= dfp[1];
            fp[j][2] -= dfp[2];
          }
        }
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 Calculate pressure from density
 ------------------------------------------------------------------------- */

double PairRHEO::calc_pressure(double rho, int type)
{
  double P = 0;

  if (pressure_style == LINEAR) {
    P = csq[type]*(rho-rho0[type]);
  } else if (pressure_style == CUBIC) {
    double dr = rho-rho0[type];
    P = csq[type]*(dr + c_pressure_cubic*dr*dr*dr);
  } else if (pressure_style == TAITWATER) {
    double rho_ratio = rho/rho0[type];
    double rr3 = rho_ratio*rho_ratio*rho_ratio;
    P = csq[type]*rho0[type]/7.0*(rr3*rr3*rho_ratio-1.0);
  }

  return P;
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairRHEO::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(rho0, n + 1, "pair:rho0");
  memory->create(csq, n + 1, "pair:csq");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRHEO::settings(int narg, char **arg)
{
  if(narg < 1) error->all(FLERR,"Illegal pair_style command");

  h = utils::numeric(FLERR,arg[0],false,lmp);
  if(h <= 0.0) error->all(FLERR,"Illegal pair_style command");

  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "laplacian") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"Illegal pair_style command");
      laplacian_order = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (laplacian_order > 2 || laplacian_order < 0) error->all(FLERR,
        "Laplacian estimator can be between zero to second order");
      iarg += 1;
    } else if (strcmp(arg[iarg], "pressure") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"Illegal pair_style command");

      if (strcmp(arg[iarg+1], "taitwater") == 0) {
        pressure_style = TAITWATER;
      } else if (strcmp(arg[iarg+1], "linear") == 0) {
        pressure_style = LINEAR;
      } else if (strcmp(arg[iarg+1], "cubic") == 0) {
        if (iarg+2 >= narg) error->all(FLERR,"Illegal pair_style command");
        pressure_style = CUBIC;
        c_pressure_cubic = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        iarg ++;
      }
      iarg ++;
    } else if (strcmp(arg[iarg], "rho/damp") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"Illegal pair_style command");

      rho_damp_flag = 1;
      rho_damp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg ++;
    } else if (strcmp(arg[iarg], "artificial/visc") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"Illegal pair_style command");

      artificial_visc_flag = 1;
      av = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg ++;
    } else error->all(FLERR,"Illegal pair_style command");
    iarg++;
  }

  if ( pressure_style != LINEAR) error->warning(FLERR, "Need linear pressure style for assumption in compute rheo/solids");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairRHEO::coeff(int narg, char **arg)
{
  if (narg != 4)
    error->all(FLERR,"Incorrect number of args for pair_style llns coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR,arg[0],1, atom->ntypes, ilo, ihi,error);
  utils::bounds(FLERR,arg[1],1, atom->ntypes, jlo, jhi,error);

  double rho0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double c_one = utils::numeric(FLERR,arg[3],false,lmp);

  if (c_one != 1.0) error->warning(FLERR, "Need c = 1 for assumption in compute rheo/solids");
  if (rho0_one != 1.0) error->warning(FLERR, "Need rho0 = 1 for assumption in compute rheo/solids");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    rho0[i] = rho0_one;
    csq[i] = c_one*c_one;

    for (int j = 0; j <= atom->ntypes; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair llns coefficients");
}

/* ----------------------------------------------------------------------
 setup specific to this pair style
 ------------------------------------------------------------------------- */

void PairRHEO::setup()
{
  int flag;
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Using pair RHEO without fix RHEO");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);

  compute_kernel = fix_rheo->compute_kernel;
  compute_grad = fix_rheo->compute_grad;
  compute_solids = fix_rheo->compute_solids;
  thermal_flag = fix_rheo->thermal_flag;

  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair RHEO requires ghost atoms store velocity");

  if (laplacian_order == -1) {
    if (fix_rheo->kernel_type == FixRHEO::CRK2)
      laplacian_order = 2;
    else if (fix_rheo->kernel_type == FixRHEO::CRK1)
      laplacian_order = 1;
    else
      laplacian_order = 0;
  }
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairRHEO::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair rheo coeffs are not set");
  }

  cut[i][j] = h;
  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairRHEO::single(int /*i*/, int /*j*/, int /*itype*/, int /*jtype*/,
    double /*rsq*/, double /*factor_coul*/, double /*factor_lj*/, double &fforce)
{
  fforce = 0.0;

  return 0.0;
}
