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

#include "atom_vec_rheo.h"

#include "atom.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecRHEO::AtomVecRHEO(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;
  forceclearflag = 1;

  atom->rho_flag = 1;
  atom->heat_flag = 1;
  atom->conductivity_flag = 1;
  atom->fp_flag = 1;
  atom->temp_flag = 1;
  atom->viscosity_flag = 1;
  atom->phase_flag = 1;
  atom->surface_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "rho drho fp temp heat conductivity viscosity phase surface";
  fields_copy = (char *) "rho drho fp temp heat conductivity viscosity phase surface";
  fields_comm = (char *) "rho temp conductivity viscosity phase surface";
  fields_comm_vel = (char *) "rho temp conductivity viscosity phase surface";
  fields_reverse = (char *) "drho fp heat";
  fields_border = (char *) "rho temp conductivity viscosity phase surface";
  fields_border_vel = (char *) "rho temp conductivity viscosity phase surface";
  fields_exchange = (char *) "rho temp phase surface";
  fields_restart = (char * ) "rho temp phase surface";
  fields_create = (char *) "rho drho fp temp heat conductivity viscosity phase surface";
  fields_data_atom = (char *) "id type rho temp phase x";
  fields_data_vel = (char *) "id v";

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecRHEO::grow_pointers()
{
  rho = atom->rho;
  drho = atom->drho;
  fp = atom->fp;
  heat = atom->heat;
  temp = atom->temp;
  conductivity = atom->conductivity;
  viscosity = atom->viscosity;
  phase = atom->phase;
  surface = atom->surface;
}

/* ----------------------------------------------------------------------
   clear extra forces starting at atom N
   nbytes = # of bytes to clear for a per-atom vector
------------------------------------------------------------------------- */

void AtomVecRHEO::force_clear(int n, size_t nbytes)
{
  memset(&fp[n][0],0,3*nbytes);
  memset(&drho[n],0,nbytes);
  memset(&heat[n],0,nbytes);
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecRHEO::create_atom_post(int ilocal)
{
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecRHEO::data_atom_post(int ilocal)
{
  fp[ilocal][0] = 0.0;
  fp[ilocal][1] = 0.0;
  fp[ilocal][2] = 0.0;
  drho[ilocal] = 0.0;
  heat[ilocal] = 0.0;
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecRHEO::property_atom(char *name)
{
  if (strcmp(name,"rho") == 0) return 0;
  if (strcmp(name,"drho") == 0) return 1;
  if (strcmp(name,"fpx") == 0) return 2;
  if (strcmp(name,"fpy") == 0) return 3;
  if (strcmp(name,"fpz") == 0) return 4;
  if (strcmp(name,"temp") == 0) return 5;
  if (strcmp(name,"heat") == 0) return 6;
  if (strcmp(name,"conductivity") == 0) return 7;
  if (strcmp(name,"viscosity") == 0) return 8;
  if (strcmp(name,"phase") == 0) return 9;
  if (strcmp(name,"surface") == 0) return 10;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecRHEO::pack_property_atom(int index, double *buf,
                                     int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = rho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = drho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 2) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = fp[i][0];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 3) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = fp[i][1];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 4) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = fp[i][2];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 5) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = temp[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 6) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = heat[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 7) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = conductivity[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 8) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = viscosity[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 9) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = phase[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 10) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = surface[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}
