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

#include "fix_rheo_inlet.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "modify.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "fix_rheo.h"
#include "compute_rheo_kernel.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace LAMMPS_NS;
using namespace FixConst;

#define MAXLINE 256
#define CHUNK 1024
#define LB_FACTOR 5.0 // set high in case have narrow inlets
#define NSECTIONS 25       // change when add to header::section_keywords
#define EPSILON 1.0e-6

/*
Todo:
  3) Allow non-modular box sizes - use 2 strips at a time and check atom positions
  4) Generalize to inserting on any box domain
  5) Error for triclinic
  7) Generalize to non-rheo cases (e.g. granular)
*/


/* ---------------------------------------------------------------------- */

FixRHEOInlet::FixRHEOInlet(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix deposit command");

  MPI_Comm_rank(world,&me);
  line = new char[MAXLINE];
  keyword = new char[MAXLINE];
  buffer = new char[CHUNK*MAXLINE];

  nmax = 0;
  vsum = nullptr;
  norm = nullptr;
  comm_reverse = 2;

  time_depend = 1;
  dynamic_group_allow = 1;

  cut = utils::numeric(FLERR,arg[3],false,lmp);
  cutsq = cut*cut;
  filename = arg[4];
  dL_insert = utils::numeric(FLERR,arg[5],false,lmp);
  vx_insert = utils::numeric(FLERR,arg[6],false,lmp);

  if (dL_insert < cut)
    error->all(FLERR,"Insert width less than cutoff");

  outlet_flag = 1;
  if (narg == 8 && strcmp(arg[7],"outlet/no") == 0)
    outlet_flag = 0;

  // error check and further setup for mode = MOLECULE

  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use fix rheo/inlet unless atoms have IDs");

  // find current max atom

  find_maxid();

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  nfirst = next_reneighbor;
  ninserted = 0;
  current_slice = 0;

  n_stored = nullptr;
  x_stored = nullptr;
  phase_stored = nullptr;
  type_stored = nullptr;
  rho_stored = nullptr;
  temp_stored = nullptr;

  fix_rheo = nullptr;
  compute_kernel = nullptr;
}

/* ---------------------------------------------------------------------- */

FixRHEOInlet::~FixRHEOInlet()
{
  memory->destroy(n_stored);
  memory->destroy(x_stored);
  memory->destroy(phase_stored);
  memory->destroy(type_stored);
  memory->destroy(rho_stored);
  memory->destroy(temp_stored);
  memory->destroy(vsum);
  memory->destroy(norm);
}

/* ---------------------------------------------------------------------- */

int FixRHEOInlet::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRHEOInlet::init()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;

  int flag;
  int ifix = modify->find_fix_by_style("rheo");
  if (ifix == -1) error->all(FLERR, "Using pair RHEO without fix RHEO");
  fix_rheo = ((FixRHEO *) modify->fix[ifix]);
  compute_kernel = fix_rheo->compute_kernel;

  if (domain->box_change)
    error->all(FLERR, "Cannot use fix rheo/inlet with changing box");

  // Skip if data file already read and system allocated
  // This assumes the user didn't change the box...
  // Also assumes no fix balance...
  if (x_stored) return;

  double dx, temp, whole, fractional;
  dx = vx_insert*update->dt;
  temp = dL_insert/dx;
  fractional = std::modf(temp, &whole);
  if (fabs(fractional) > EPSILON) error->all(FLERR, "Velocity incompatible with slice size");
  time_insert = (int) whole;

  buffer = new char[CHUNK*MAXLINE];
  natoms = 0;
  keyword[0] = '\0';
  me = comm->me;

  while (1) {

    // open file on proc 0

    if (me == 0) {
      utils::logmesg(lmp,"Reading data file ...\n");
      open(filename);
    } else fp = nullptr;

    // read header info

    header();

    // Box dimensions must match data file
    if (domain->boxlo[1] != boxlo[1] || domain->boxhi[1] != boxhi[1] ||
      domain->boxlo[2] != boxlo[2] || domain->boxhi[2] != boxhi[2])
      error->all(FLERR, "Box bounds do not match data file");

    // Estimate how many particles are in each strip
    // assuming homogeneous density, conservatively round down
    double L = boxhi[0]-boxlo[0];
    nstrips = floor(L/dL_insert);

    // Ensure box is evenly distributed by this number of strips
    // BUT must be smaller so all atoms can be assigned
    if ((nstrips*dL_insert - L) > 0.0)
      error->all(FLERR, "Data file box larger than current box");
    if ((nstrips*dL_insert - L) < -0.01)
      error->all(FLERR, "Data file box not evenly divided by dL");

    // Divide them up based on how processors are on that boundary
    int nprocs = comm->procgrid[1]*comm->procgrid[2];
    natoms_max_insert = static_cast<int> (LB_FACTOR * natoms / nprocs / nstrips);

    memory->create(n_stored, nstrips, "fix_rheo_inlet:n_stored");
    memory->create(x_stored, nstrips, natoms_max_insert, 3, "fix_rheo_inlet:x_stored");
    memory->create(phase_stored, nstrips, natoms_max_insert,"fix_rheo_inlet:phase_stored");
    memory->create(rho_stored, nstrips, natoms_max_insert, "fix_rheo_inlet:rho_stored");
    memory->create(type_stored, nstrips, natoms_max_insert,"fix_rheo_inlet:type_stored");
    memory->create(temp_stored, nstrips, natoms_max_insert,"fix_rheo_inlet:temp_stored");

    for (int n = 0; n < nstrips; n++) n_stored[n] = 0;

    // customize for new sections -only support a few
    // read rest of file in free format

    while (strlen(keyword)) {
      if (strcmp(keyword,"Atoms") == 0) {
        atoms();
      } else if (strcmp(keyword,"Velocities") == 0) {
        skip_lines(natoms);
      } else if (strcmp(keyword,"Masses") == 0) {
        skip_lines(ntypes);
      } else if (strcmp(keyword,"Pair Coeffs") == 0) {
        skip_lines(ntypes);
      } else {
        error->all(FLERR, "Bad keyword in datafile, only accept atom, velocity, mass, pair coeff");
      }
      parse_keyword(0);
    }

    // close file

    if (me == 0) {
      fclose(fp);
      fp = nullptr;
    }
    break;
  }
  current_slice = nstrips-1;
  delete [] buffer;
}

/* ---------------------------------------------------------------------- */

void FixRHEOInlet::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   perform particle insertion
------------------------------------------------------------------------- */

void FixRHEOInlet::pre_exchange()
{
  int i,m,n,itype;
  double coord[3];

  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;
  // clear ghost count and any ghost bonus data internal to AtomVec
  // same logic as beginning of Comm::exchange()
  // do it now b/c inserting atoms will overwrite ghost atoms

  atom->nghost = 0;
  int nlocal_previous = atom->nlocal;
  atom->avec->clear_bonus();

  // find current max atom and molecule IDs if necessary

  find_maxid();

  // sum atoms inserted on all processors with a lower ID than me
  int maxtag_me = maxtag_all;
  int temp;
  for (m = 0; m < comm->nprocs; m++) {
    temp = n_stored[current_slice];
    MPI_Bcast(&temp, 1, MPI_INT, m, world);
    if (m < comm->me) maxtag_me += temp;
  }

  // Check if any particles in deposition area to warn
  tagint *tag = atom->tag;

  for (m = 0; m < atom->nlocal; m++) {
    if (atom->x[m][0] < dL_insert) {
      printf("Atom %d at %g\n", atom->tag[m], atom->x[m][0]);
      error->warning(FLERR, "Atom still in deposition zone");
    }
  }

  // Add atoms in the current slice

  for (m = 0; m < n_stored[current_slice]; m++) {
    coord[0] = x_stored[current_slice][m][0];
    coord[1] = x_stored[current_slice][m][1];
    coord[2] = x_stored[current_slice][m][2];
    itype = type_stored[current_slice][m];
    atom->avec->create_atom(itype,coord);

    n = atom->nlocal-1;
    atom->rho[n] = rho_stored[current_slice][m];
    atom->temp[n] = temp_stored[current_slice][m];
    atom->phase[n] = phase_stored[current_slice][m];
    atom->tag[n] = maxtag_me + m+1;

    atom->v[n][0] = vx_insert;
    atom->v[n][1] = 0.0;
    atom->v[n][2] = 0.0;
  }

  // reset global natoms,nbonds,etc
  // increment maxtag_all and maxmol_all if necessary
  // if global map exists, reset it now instead of waiting for comm
  //   since other pre-exchange fixes may use it
  //   invoke map_init() b/c atom count has grown

  atom->natoms += n_stored[current_slice];
  if (atom->natoms < 0)
    error->all(FLERR,"Too many total atoms");

  maxtag_all += n_stored[current_slice];
  if (maxtag_all >= MAXTAGINT)
    error->all(FLERR,"New atom IDs exceed maximum allowed ID");

  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }

  atom->data_fix_compute_variable(nlocal_previous,atom->nlocal);
  // next timestep to insert
  // next_reneighbor = 0 if done

  current_slice -= 1;
  if (current_slice == -1) current_slice = nstrips-1;
  next_reneighbor += time_insert;
}

/* ----------------------------------------------------------------------
   remove forces and drho from particles in region
------------------------------------------------------------------------- */

void FixRHEOInlet::post_force(int /*vflag*/)
{
  int i;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **fp = atom->fp;
  double *heat = atom->heat;
  double *drho = atom->drho;
  double *rho = atom->rho;
  int *phase = atom->phase;
  int *mask = atom->mask;

  for (i = 0; i < atom->nlocal; i ++) {
    if (x[i][0] < 2*dL_insert && mask[i] & groupbit) { // Add a buffer 2x as large
      // Solids only 1/2 buffer since they have fewer
      // issues with density etc (also walls don't move)
      if (x[i][0] >= dL_insert && phase[i] > FixRHEO::FLUID_MAX) continue;
      v[i][0] = vx_insert;
      v[i][1] = 0.0;
      v[i][2] = 0.0;
      heat[i] = 0.0;
      if (phase[i] <= FixRHEO::FLUID_MAX) phase[i] = FixRHEO::FLUID_NO_FORCE;
    }
  }


  if (outlet_flag) {
    if (nmax < atom->nmax) {
      nmax = atom->nmax;
      memory->grow(vsum, nmax, "inlet/rho0");
      memory->grow(norm, nmax, "inlet/save0");
    }

    // neighbor list variables
    int *jlist;
    int inum, *ilist, *numneigh, **firstneigh;
    int nlocal = atom->nlocal;
    int nall = nlocal + atom->nghost;
    int newton_pair = force->newton_pair;
    int j, a, b, ii, jj, jnum;
    double w,xi, xj, ytmp, ztmp, r, rsq;
    double dx[3];
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    double xhi = domain->boxhi[0];

    for (i = 0; i < nall; i++) {
      w = compute_kernel->calc_w(i, i, 0,0,0,0);
      vsum[i] = w*v[i][0];
      norm[i] = w;
    }

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xi = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];

      if (xi < (xhi - 2*cut)) continue;

      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        xj = x[j][0];
        if (xj < (xhi - 2*cut)) continue;
        if (xi < (xhi-cut) && xj < (xhi-cut)) continue;
        if (xi >= (xhi-cut*0.5) && xj >= (xhi-cut*0.5)) continue;

        dx[0] = xi - xj;
        dx[1] = ytmp - x[j][1];
        dx[2] = ztmp - x[j][2];
        rsq = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];

        if (rsq < cutsq) {
          r = sqrt(rsq);
          w = compute_kernel->calc_w(i, j, dx[0], dx[1], dx[2], r);
          if (xi >= xhi-cut) {
            vsum[i] += w*v[j][0];
            norm[i] += w;
          }

          if (newton_pair || j < nlocal) {
            if (xj >= xhi-cut) {
              vsum[j] += w*v[i][0];
              norm[j] += w;
            }
          }
        }
      }
    }
    if (newton_pair) comm->reverse_comm_fix(this);


    // Outlet condition - assign average x velocity to fluid
    double sm1, sm2;
    for (int i = 0; i < atom->nlocal; i ++) {
      if (x[i][0] >= (xhi-cut) && mask[i] & groupbit) {
        if (phase[i] > FixRHEO::FLUID_MAX) continue;
        if (norm[i] != 0.0) v[i][0] = vsum[i]/norm[i];
        else v[i][0] = 0.0;

        v[i][1] = 0.0;
        v[i][2] = 0.0;
        phase[i] = FixRHEO::FLUID_NO_FORCE;
      }
    }
  }
}


/* ---------------------------------------------------------------------- */

int FixRHEOInlet::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = vsum[i];
    buf[m++] = norm[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRHEOInlet::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    vsum[j] += buf[m++];
    norm[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
------------------------------------------------------------------------- */

void FixRHEOInlet::find_maxid()
{
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
}


/* ----------------------------------------------------------------------
   read free-format header of data file
   1st line and blank lines are skipped
   non-blank lines are checked for header keywords and leading value is read
   header ends with EOF or non-blank line containing no header keyword
     if EOF, line is set to blank line
     else line has first keyword line for rest of file
   some logic differs if adding atoms
------------------------------------------------------------------------- */

void FixRHEOInlet::header()
{
  int n;
  char *ptr;

  // customize for new sections

  const char *section_keywords[NSECTIONS] =
    {"Atoms","Velocities","Ellipsoids","Lines","Triangles","Bodies",
     "Bonds","Angles","Dihedrals","Impropers",
     "Masses","Pair Coeffs","PairIJ Coeffs","Bond Coeffs","Angle Coeffs",
     "Dihedral Coeffs","Improper Coeffs",
     "BondBond Coeffs","BondAngle Coeffs","MiddleBondTorsion Coeffs",
     "EndBondTorsion Coeffs","AngleTorsion Coeffs",
     "AngleAngleTorsion Coeffs","BondBond13 Coeffs","AngleAngle Coeffs"};

  // skip 1st line of file

  if (me == 0) {
    char *eof = fgets(line,MAXLINE,fp);
    if (eof == nullptr) error->one(FLERR,"Unexpected end of data file");
  }

  while (1) {

    // read a line and bcast length

    if (me == 0) {
      if (fgets(line,MAXLINE,fp) == nullptr) n = 0;
      else n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,world);

    // if n = 0 then end-of-file so return with blank line

    if (n == 0) {
      line[0] = '\0';
      return;
    }

    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // trim anything from '#' onward
    // if line is blank, continue

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    if (strspn(line," \t\n\r") == strlen(line)) continue;

    // search line for atom, atom types, and box header keyword and set corresponding variable
    // ignore all other headers
    int rv;
    if (utils::strmatch(line,"^\\s*\\d+\\s+atoms\\s")) {
      rv = sscanf(line,BIGINT_FORMAT,&natoms);
      if (rv != 1)
        error->all(FLERR,"Could not parse 'atoms' line in data file header");
      if (natoms < 0 || natoms >= MAXBIGINT)
        error->all(FLERR,"System in data file is too big");
    } else if (utils::strmatch(line,"^\\s*\\d+\\s+atom\\s+types\\s")) {
      rv = sscanf(line,"%d",&ntypes);
      if (rv != 1)
        error->all(FLERR,"Could not parse 'atom types' line "
                   "in data file header");
      if (ntypes != atom->ntypes)
        error->all(FLERR,"Number of atom types in data file does not match");

    // local copy of box info
    // so can treat differently for first vs subsequent data files

    } else if (utils::strmatch(line,"^\\s*\\f+\\s+\\f+\\s+xlo\\s+xhi\\s")) {
      rv = sscanf(line,"%lg %lg",&boxlo[0],&boxhi[0]);
      if (rv != 2)
        error->all(FLERR,"Could not parse 'xlo xhi' line in data file header");

    } else if (utils::strmatch(line,"^\\s*\\f+\\s+\\f+\\s+ylo\\s+yhi\\s")) {
      rv = sscanf(line,"%lg %lg",&boxlo[1],&boxhi[1]);
      if (rv != 2)
        error->all(FLERR,"Could not parse 'ylo yhi' line in data file header");

    } else if (utils::strmatch(line,"^\\s*\\f+\\s+\\f+\\s+zlo\\s+zhi\\s")) {
      rv = sscanf(line,"%lg %lg",&boxlo[2],&boxhi[2]);
      if (rv != 2)
        error->all(FLERR,"Could not parse 'zlo zhi' line in data file header");

    } else if (utils::strmatch(line,"^\\s*\\f+\\s+\\f+\\s+\\f+"
                               "\\s+xy\\s+xz\\s+yz\\s")) {
      error->all(FLERR,"Cannot use triclinic box with fix rheo/inlet");

    } else break;
  }

  // check that exiting string is a valid section keyword

  parse_keyword(1);
  for (n = 0; n < NSECTIONS; n++)
    if (strcmp(keyword,section_keywords[n]) == 0) break;
  if (n == NSECTIONS)
    error->all(FLERR,fmt::format("Unknown identifier in data file: {}",keyword));
}

/* ----------------------------------------------------------------------
   proc 0 opens data file, can't be gzipped
------------------------------------------------------------------------- */

void FixRHEOInlet::open(std::string file)
{
  fp = fopen(file.c_str(),"r");

  if (fp == nullptr)
    error->one(FLERR,fmt::format("Cannot open file {}: {}",
                                 file, utils::getsyserror()));
}


/* ----------------------------------------------------------------------
   grab next keyword
   read lines until one is non-blank
   keyword is all text on line w/out leading & trailing white space
   optional style can be appended after comment char '#'
   read one additional line (assumed blank)
   if any read hits EOF, set keyword to empty
   if first = 1, line variable holds non-blank line that ended header
------------------------------------------------------------------------- */

void FixRHEOInlet::parse_keyword(int first)
{
  int eof = 0;
  int done = 0;

  // proc 0 reads upto non-blank line plus 1 following line
  // eof is set to 1 if any read hits end-of-file

  if (me == 0) {
    if (!first) {
      if (fgets(line,MAXLINE,fp) == nullptr) eof = 1;
    }
    while (eof == 0 && done == 0) {
      int blank = strspn(line," \t\n\r");
      if ((blank == (int)strlen(line)) || (line[blank] == '#')) {
        if (fgets(line,MAXLINE,fp) == nullptr) eof = 1;
      } else done = 1;
    }
    if (fgets(buffer,MAXLINE,fp) == nullptr) {
      eof = 1;
      buffer[0] = '\0';
    }
  }

  // if eof, set keyword empty and return

  MPI_Bcast(&eof,1,MPI_INT,0,world);
  if (eof) {
    keyword[0] = '\0';
    return;
  }

  // bcast keyword line to all procs

  int n;
  if (me == 0) n = strlen(line) + 1;
  MPI_Bcast(&n,1,MPI_INT,0,world);
  MPI_Bcast(line,n,MPI_CHAR,0,world);

  // store optional "style" following comment char '#' after keyword

  char *ptr;
  if ((ptr = strchr(line,'#'))) {
    *ptr++ = '\0';
    while (*ptr == ' ' || *ptr == '\t') ptr++;
    int stop = strlen(ptr) - 1;
    while (ptr[stop] == ' ' || ptr[stop] == '\t'
           || ptr[stop] == '\n' || ptr[stop] == '\r') stop--;
    ptr[stop+1] = '\0';
    strcpy(style,ptr);
  } else style[0] = '\0';

  // copy non-whitespace portion of line into keyword

  int start = strspn(line," \t\n\r");
  int stop = strlen(line) - 1;
  while (line[stop] == ' ' || line[stop] == '\t'
         || line[stop] == '\n' || line[stop] == '\r') stop--;
  line[stop+1] = '\0';
  strcpy(keyword,&line[start]);
}

/* ----------------------------------------------------------------------
   proc 0 reads N lines from file
   could be skipping Natoms lines, so use bigints
------------------------------------------------------------------------- */

void FixRHEOInlet::skip_lines(bigint n)
{
  if (me) return;
  if (n <= 0) return;
  char *eof = nullptr;
  for (bigint i = 0; i < n; i++) eof = fgets(line,MAXLINE,fp);
  if (eof == nullptr) error->one(FLERR,"Unexpected end of data file");
}


/* ----------------------------------------------------------------------
   read all atoms
------------------------------------------------------------------------- */

void FixRHEOInlet::atoms()
{
  int nchunk,eof;
  double shift[3];
  shift[0] = shift[1] = shift[2] = 0.0;

  if (me == 0) utils::logmesg(lmp,"  1 reading atoms ...\n");

  bigint nread = 0;
  while (nread < natoms) {
    nchunk = MIN(natoms-nread,CHUNK);
    eof = comm->read_lines_from_file(fp,nchunk,MAXLINE,buffer);
    if (eof) error->all(FLERR,"Unexpected end of data file");
    data_atoms(nchunk,buffer,0,0,0,0,shift);
    nread += nchunk;
  }
}

/* ----------------------------------------------------------------------
   unpack N lines from Atom section of data file
   call rheo-specific routine to parse line
------------------------------------------------------------------------- */

void FixRHEOInlet::data_atoms(int n, char *buf, tagint id_offset, tagint mol_offset,
                      int type_offset, int shiftflag, double *shift)
{
  int m,xptr,iptr;
  imageint imagedata;
  double xdata[3], epsilon[3];
  double *coord;
  char *next;

  next = strchr(buf,'\n');
  *next = '\0';
  int nwords = utils::trim_and_count_words(buf);
  *next = '\n';

  if (nwords != atom->avec->size_data_atom && nwords != atom->avec->size_data_atom + 3)
    error->all(FLERR,"Incorrect atom format in data file");

  char **values = new char*[nwords];

  // set bounds for my proc
  // if periodic and I am lo/hi proc, adjust bounds by EPSILON
  // insures all data atoms will be owned even with round-off

  epsilon[0] = domain->prd[0] * EPSILON;
  epsilon[1] = domain->prd[1] * EPSILON;
  epsilon[2] = domain->prd[2] * EPSILON;

  double sublo[3],subhi[3];
  sublo[0] = domain->sublo[0]; subhi[0] = domain->subhi[0];
  sublo[1] = domain->sublo[1]; subhi[1] = domain->subhi[1];
  sublo[2] = domain->sublo[2]; subhi[2] = domain->subhi[2];


  if (comm->layout != Comm::LAYOUT_TILED) {
    if (domain->xperiodic) {
      if (comm->myloc[0] == 0) sublo[0] -= epsilon[0];
      if (comm->myloc[0] == comm->procgrid[0]-1) subhi[0] += epsilon[0];
    }
    if (domain->yperiodic) {
      if (comm->myloc[1] == 0) sublo[1] -= epsilon[1];
      if (comm->myloc[1] == comm->procgrid[1]-1) subhi[1] += epsilon[1];
    }
    if (domain->zperiodic) {
      if (comm->myloc[2] == 0) sublo[2] -= epsilon[2];
      if (comm->myloc[2] == comm->procgrid[2]-1) subhi[2] += epsilon[2];
    }
  } else {
    error->all(FLERR, "Cannot use tiled comm layout");
  }

  // Skip for processors not on the boundary
  if (comm->myloc[0] != 0) return;

  // xptr = which word in line starts xyz coords

  xptr = atom->avec->xcol_data - 1;
  int imageflag = 0;
  if (nwords > atom->avec->size_data_atom) imageflag = 1;
  if (imageflag) iptr = nwords - 3;

  // loop over lines of atom data
  // tokenize the line into values
  // extract xyz coords and image flags
  // remap atom into simulation box
  // if atom is in my sub-domain, unpack its values

  int flagx = 0, flagy = 0, flagz = 0;
  for (int i = 0; i < n; i++) {
    next = strchr(buf,'\n');

    values[0] = strtok(buf," \t\n\r\f");
    if (values[0] == nullptr)
      error->all(FLERR,"Incorrect atom format in data file");
    for (m = 1; m < nwords; m++) {
      values[m] = strtok(nullptr," \t\n\r\f");
      if (values[m] == nullptr)
        error->all(FLERR,"Incorrect atom format in data file");
    }

    int imx = 0, imy = 0, imz = 0;
    if (imageflag) {
      imy = utils::inumeric(FLERR,values[iptr+1],false,lmp);
      imz = utils::inumeric(FLERR,values[iptr+2],false,lmp);
      if ((domain->dimension == 2) && (imz != 0))
        error->all(FLERR,"Z-direction image flag must be 0 for 2d-systems");
      if ((!domain->yperiodic) && (imy != 0)) flagy = 1;
      if ((!domain->zperiodic) && (imz != 0)) flagz = 1;
    }
    imagedata = ((imageint) (imx + IMGMAX) & IMGMASK) |
        (((imageint) (imy + IMGMAX) & IMGMASK) << IMGBITS) |
        (((imageint) (imz + IMGMAX) & IMGMASK) << IMG2BITS);

    xdata[0] = utils::numeric(FLERR,values[xptr],false,lmp);
    xdata[1] = utils::numeric(FLERR,values[xptr+1],false,lmp);
    xdata[2] = utils::numeric(FLERR,values[xptr+2],false,lmp);

    domain->remap(xdata,imagedata);
    coord = xdata;

    int strip_id = coord[0]/dL_insert;
    double xtemp = coord[0] - strip_id*dL_insert;

    if (xtemp >= sublo[0]    && xtemp < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) {
      data_atom(xdata,imagedata,values);
    }

    buf = next + 1;
  }

  // warn if reading data with non-zero image flags for non-periodic boundaries.
  // we may want to turn this into an error at some point, since this essentially
  // creates invalid position information that works by accident most of the time.

  if (comm->me == 0) {
    if (flagy)
      error->warning(FLERR,"Non-zero imageflag(s) in y direction for non-periodic boundary");
    if (flagz)
      error->warning(FLERR,"Non-zero imageflag(s) in z direction for non-periodic boundary");
  }

  delete [] values;
}


/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
------------------------------------------------------------------------- */

void FixRHEOInlet::data_atom(double *coord, imageint imagetmp, char **values)
{
  int m,n,datatype,cols;
  void *pdata;
  double L = boxhi[0]-boxlo[0];
  int strip_id = coord[0]/dL_insert;
  int atom_id = n_stored[strip_id];

  n_stored[strip_id] += 1;
  if (n_stored[strip_id] > natoms_max_insert) {
    printf("nstored = %d vs %d\n", n_stored[strip_id], natoms_max_insert);
    error->one(FLERR, "Strip memory overflow");
  }

  x_stored[strip_id][atom_id][0] = coord[0] - strip_id*dL_insert;
  x_stored[strip_id][atom_id][1] = coord[1];
  x_stored[strip_id][atom_id][2] = coord[2];

  type_stored[strip_id][atom_id] = utils::inumeric(FLERR,values[1],true,lmp);
  rho_stored[strip_id][atom_id] = utils::numeric(FLERR,values[2],true,lmp);
  temp_stored[strip_id][atom_id] = utils::numeric(FLERR,values[3],true,lmp);
  phase_stored[strip_id][atom_id] = utils::inumeric(FLERR,values[4],true,lmp);
  if (phase_stored[strip_id][atom_id] == FixRHEO::FLUID) phase_stored[strip_id][atom_id] = FixRHEO::FLUID_NO_SHIFT;

}
