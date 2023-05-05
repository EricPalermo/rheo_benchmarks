/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(rheo/inlet,FixRHEOInlet)

#else

#ifndef LMP_FIX_RHEO_INLET_H
#define LMP_FIX_RHEO_INLET_H

#include "fix.h"
#include <string>

namespace LAMMPS_NS {

class FixRHEOInlet : public Fix {
 public:
  FixRHEOInlet(class LAMMPS *, int, char **);
  ~FixRHEOInlet();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void pre_exchange();
  void post_force(int);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
 private:
  int nmax;
  double cut, cutsq, *norm, *vsum;

  int outlet_flag;

  int *n_stored;
  double ***x_stored;
  int **phase_stored, **type_stored;
  double **rho_stored, **temp_stored;
  double dL_insert, vx_insert;
  int time_insert;
  int current_slice;

  std::string filename;
  tagint maxtag_all;
  int ninserted;
  int nfirst;

  int me,compressed;
  FILE *fp;
  char *line,*keyword,*buffer;
  bigint natoms;
  bigint natoms_max_insert;
  int nstrips, ntypes;
  double boxlo[3],boxhi[3];

  void open(std::string);
  void header();
  void parse_keyword(int);
  void skip_lines(bigint);
  void atoms();
  void data_atoms(int, char *, tagint, tagint, int, int, double *);
  virtual void data_atom(double *, imageint, char **);
  void find_maxid();

  class ComputeRHEOKernel *compute_kernel;
  class FixRHEO *fix_rheo;
  class NeighList *list;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix deposit command

Self-explanatory.

E: Must specify a region in fix deposit

The region keyword must be specified with this fix.

E: Fix deposit region does not support a bounding box

Not all regions represent bounded volumes.  You cannot use
such a region with the fix deposit command.

E: Fix deposit region cannot be dynamic

Only static regions can be used with fix deposit.

E: Deposition region extends outside simulation box

Self-explanatory.

E: Cannot use fix_deposit unless atoms have IDs

Self-explanatory.

E: Fix deposit molecule must have coordinates

The defined molecule does not specify coordinates.

E: Fix deposit molecule must have atom types

The defined molecule does not specify atom types.

E: Invalid atom type in fix deposit mol command

The atom types in the defined molecule are added to the value
specified in the create_atoms command, as an offset.  The final value
for each atom must be between 1 to N, where N is the number of atom
types.

E: Fix deposit molecule template ID must be same as atom_style template ID

When using atom_style template, you cannot deposit molecules that are
not in that template.

E: Cannot use fix deposit rigid and not molecule

Self-explanatory.

E: Cannot use fix deposit shake and not molecule

Self-explanatory.

E: Cannot use fix deposit rigid and shake

These two attributes are conflicting.

E: Region ID for fix deposit does not exist

Self-explanatory.

E: Fix deposit rigid fix does not exist

UNDOCUMENTED

E: Fix deposit and fix rigid/small not using same molecule template ID

Self-explanatory.

E: Fix deposit shake fix does not exist

Self-explanatory.

E: Fix deposit and fix shake not using same molecule template ID

Self-explanatory.

W: Fix deposit near setting < possible overlap separation %g

This test is performed for finite size particles with a diameter, not
for point particles.  The near setting is smaller than the particle
diameter which can lead to overlaps.

E: Unknown particle distribution in fix deposit

UNDOCUMENTED

W: Particle deposition was unsuccessful

The fix deposit command was not able to insert as many atoms as
needed.  The requested volume fraction may be too high, or other atoms
may be in the insertion region.

E: Too many total atoms

See the setting for bigint in the src/lmptype.h file.

E: New atom IDs exceed maximum allowed ID

See the setting for tagint in the src/lmptype.h file.

E: Molecule template ID for fix deposit does not exist

Self-explanatory.

U: Fix pour rigid fix does not exist

Self-explanatory.

*/
