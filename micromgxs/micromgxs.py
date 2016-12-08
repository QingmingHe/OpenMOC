#!/usr/bin/env python
# -*- coding: utf-8 -*-
# todo:
# sab for H in H2O
# energy cut off
import xml.etree.ElementTree as ET
from potentials import average_potentials
from goldstein_cohen import average_lambda
import openmc
import os
import numpy as np
from glob import glob
import h5py
from math import pi
import re
import subprocess
from time import sleep
from openmoc import SDSolver
from openmoc.library_ce import LibraryCe

RESONANCE_FISSION_AUTO = 1
RESONANCE_FISSION_USER = 2
DEFAULT_BATCHES = 10
DEFAULT_INACTIVE = 5
DEFAULT_PARTICLES = 100
p = subprocess.Popen('hostname', shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
_hostname = p.communicate()[0].strip()
_cross_sections = os.getenv('OPENMC_CROSS_SECTIONS')
if _cross_sections is None:
    raise Exception('OPENMC_CROSS_SECTIONS env var should be set!')


def _execute_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    return p.communicate()[0]


def _wait_finished(jobid):
    while True:
        sleep(5)
        out = _execute_command('squeue')
        jobids = []
        for aline in out.split(b'\n')[1:]:
            if len(aline) > 0:
                jobids.append(aline.strip().split()[0])
        if jobid not in jobids:
            break


def _run_openmc():
    global _hostname
    if _hostname == b'kilkenny':
        out = _execute_command('run_openmc')
        jobid = out.strip().split()[-1]
        print('running job %s ...' % (jobid))
        _wait_finished(jobid)
    else:
        openmc.run()


def set_default_settings(batches=None, inactive=None, particles=None):
    global DEFAULT_BATCHES, DEFAULT_INACTIVE, DEFAULT_PARTICLES
    if batches is not None:
        DEFAULT_BATCHES = batches
    if inactive is not None:
        DEFAULT_INACTIVE = inactive
    if particles is not None:
        DEFAULT_PARTICLES = particles


def _get_potentials_from_endf(endf_path, potentials_fname):
    files = glob(os.path.join(endf_path, "*"))
    potentials = {}
    names = []
    for afile in sorted(files):
        name = _endf_fname_to_name(afile)
        potential = _get_potential_from_endf(afile)
        potentials[name] = potential
        names.append(name)

    with open(potentials_fname, 'w') as f:
        for name in names:
            f.write('\"%s\": %f,\n' % (name, potentials[name]))


def _endf_fname_to_name(fname):
    basename = os.path.basename(fname)
    names = re.findall("n-\d+-([a-zA-Z]{1,2})-(\d+)([mM])?", basename)[0]
    name = names[0] + str(int(names[1]))
    if len(names[2]) != 0:
        name += "_m1"
    return name


def _float_fortran(string):
    x, y = re.split('[+-]', string)
    if '-' in string:
        return float(x) * 10 ** -float(y)
    else:
        return float(x) * 10 ** float(y)


def _get_potential_from_endf(fname):
    with open(fname) as f:
        for aline in f:
            if aline[71:75] == b'2151':
                f.next()
                f.next()
                aline = f.next()
                if aline[12:13] == b' ':
                    return 0.0
                else:
                    a = _float_fortran(aline.strip().split()[1])
                    return 4.0 * pi * a ** 2


def _has_resfis(A, Z):
    has_resfis = False
    if Z == 92 and (A == 233 or A == 235):
        has_resfis = True
    elif Z == 94 and (A == 239 or A == 241):
        has_resfis = True
    return has_resfis


def _get_A_Z_awr(cross_sections, materials):
    tree = ET.parse(cross_sections)
    root = tree.getroot()
    direc = os.path.dirname(cross_sections)
    for child in root:
        if materials == child.attrib['materials']:
            path = os.path.join(direc, child.attrib['path'])
            f = h5py.File(path, 'r')
            A = f[materials].attrs['A']
            Z = f[materials].attrs['Z']
            awr = f[materials].attrs['atomic_weight_ratio']
            f.close()
            return A, Z, awr
    raise Exception('%s cannot be found in %s!' % (materials, cross_sections))


def _condense_scatter(x, ig):
    ng = len(x)
    for ig0, val in enumerate(x):
        if val != 0.0:
            break
    for ig1, val in enumerate(x[::-1]):
        if val != 0.0:
            break
    ig1 = ng - ig1
    if ig0 > ig1:
        return ig, ig+1, np.array([0.0])
    else:
        return ig0, ig1, x[ig0:ig1]


class MicroMgXsOptions(object):

    def __init__(self):
        self._nuclide = None

        # Set default settings
        self._legendre_order = 1
        self._reference_dilution = 1e10
        self._fission_nuclide = 'U235'
        self._fisnuc_refdil = 800.0
        self._dilutions = [5.0, 1e1, 15.0, 20.0, 25.0, 28.0, 30.0, 35.0, 40.0,
                           45.0, 50.0, 52.0, 60.0, 70.0, 80.0, 1e2, 2e2, 4e2,
                           6e2, 1e3, 1.2e3, 1e4, 1e10]
        self._temperatures = [293.6, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
        self._group_structure = GroupStructure('wims69e')
        self._slowdown_nuclide = 'H1'
        self._background_nuclide = 'H1b'
        self._batches = DEFAULT_BATCHES
        self._inactive = DEFAULT_INACTIVE
        self._particles = DEFAULT_PARTICLES
        self._resfis_method = RESONANCE_FISSION_AUTO
        self._has_res = False
        self._has_resfis = False
        self._ri_use_openmc = True
        self._find_nearest_temp = False
        self._nu = None

    @property
    def nuclide(self):
        return self._nuclide

    @nuclide.setter
    def nuclide(self, nuclide):
        self._nuclide = nuclide

    @property
    def legendre_order(self):
        return self._legendre_order

    @legendre_order.setter
    def legendre_order(self, legendre_order):
        self._legendre_order = legendre_order

    @property
    def reference_dilution(self):
        return self._reference_dilution

    @reference_dilution.setter
    def reference_dilution(self, reference_dilution):
        self._reference_dilution = reference_dilution

    @property
    def fission_nuclide(self):
        return self._fission_nuclide

    @fission_nuclide.setter
    def fission_nuclide(self, fission_nuclide):
        self._fission_nuclide = fission_nuclide

    @property
    def slowdown_nuclide(self):
        return self._slowdown_nuclide

    @slowdown_nuclide.setter
    def slowdown_nuclide(self, slowdown_nuclide):
        self._slowdown_nuclide = slowdown_nuclide

    @property
    def fisnuc_refdil(self):
        return self._fisnuc_refdil

    @fisnuc_refdil.setter
    def fisnuc_refdil(self, fisnuc_refdil):
        self._fisnuc_refdil = fisnuc_refdil

    @property
    def dilutions(self):
        return self._dilutions

    @dilutions.setter
    def dilutions(self, dilutions):
        self._dilutions = dilutions

    @property
    def temperatures(self):
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        self._temperatures = temperatures

    @property
    def group_structure(self):
        return self._group_structure

    @group_structure.setter
    def group_structure(self, group_structure):
        self._group_structure = group_structure

    @property
    def background_nuclide(self):
        return self._background_nuclide

    @background_nuclide.setter
    def background_nuclide(self, background_nuclide):
        self._background_nuclide = background_nuclide

    @property
    def batches(self):
        return self._batches

    @batches.setter
    def batches(self, batches):
        self._batches = batches

    @property
    def inactive(self):
        return self._inactive

    @inactive.setter
    def inactive(self, inactive):
        self._inactive = inactive

    @property
    def particles(self):
        return self._particles

    @particles.setter
    def particles(self, particles):
        self._particles = particles

    @property
    def resfis_method(self):
        return self._resfis_method

    @resfis_method.setter
    def resfis_method(self, resfis_method):
        self._resfis_method = resfis_method

    @property
    def has_res(self):
        return self._has_res

    @has_res.setter
    def has_res(self, has_res):
        self._has_res = has_res

    @property
    def has_resfis(self):
        return self._has_resfis

    @has_resfis.setter
    def has_resfis(self, has_resfis):
        self._has_resfis = has_resfis

    @property
    def ri_use_openmc(self):
        return self._ri_use_openmc

    @ri_use_openmc.setter
    def ri_use_openmc(self, ri_use_openmc):
        self._ri_use_openmc = ri_use_openmc

    @property
    def find_nearest_temp(self):
        return self._find_nearest_temp

    @find_nearest_temp.setter
    def find_nearest_temp(self, find_nearest_temp):
        self._find_nearest_temp = find_nearest_temp

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, nu):
        self._nu = nu


class MicroMgXsLibrary(object):

    def __init__(self, opts_list, fname):
        # Check whether group structures are the same
        for opts in opts_list[1:]:
            if not opts.group_structure.is_same_as(
                    opts_list[0].group_structure):
                raise Exception('group structure is not the same!')
        self._opts_list = opts_list
        self._fname = fname

    def build_library_h5(self):
        f = h5py.File(self._fname)

        # Export group structure
        dset = '/group_structure'
        if dset in f:
            del f[dset]
        f[dset] = self._opts_list[0].group_structure.group_boundaries
        f[dset].attrs['first_res'] \
            = self._opts_list[0].group_structure.first_res
        f[dset].attrs['last_res'] \
            = self._opts_list[0].group_structure.last_res

        # Export average fission spectrum
        dset = '/fission_spectrum'
        if dset in f:
            del f[dset]
        f[dset] = self._opts_list[0].group_structure.fispec

        f.close()

        # Export micro mg xs for nuclides
        for opts in self._opts_list:
            xsnuc = MicroMgXsNuclide(opts)
            xsnuc.build_library()
            xsnuc.export_to_h5(self._fname)


class MicroMgXsNuclide(object):

    def __init__(self, opts):
        self._opts = opts

    def build_library(self):
        global _cross_sections

        print('processing %s ...' % (self._opts.nuclide))
        # Build full xs part
        self._full_xs = FullXs(self._opts)
        self._full_xs.build_library()

        # Set nu which may be used by RITable
        if self._opts.has_resfis:
            ig0 = self._opts.group_structure.first_res
            ig1 = self._opts.group_structure.last_res
            self._opts.nu = self._full_xs._nu_fission[0, ig0:ig1] /\
                self._full_xs._fission[0, ig0:ig1]

        # Build RI table part
        self._ri_table = RItable(self._opts)
        self._ri_table.build_library()

        # Obain A, Z and awr
        self._A, self._Z, self._awr \
            = _get_A_Z_awr(_cross_sections, self._opts._nuclide)

    def export_to_h5(self, fname):
        f = h5py.File(fname)
        root_group = '/'
        group = root_group + self._opts._nuclide

        # Create group for the nuclide
        if group in f:
            del f[group]
        f.create_group(group)

        # This is a nuclide
        f[group].attrs['is_nuclide'] = 1

        # Export header data
        f[group].attrs['A'] = self._A
        f[group].attrs['Z'] = self._Z
        f[group].attrs['awr'] = self._awr
        f[group].attrs['ng'] = self._opts.group_structure.ng
        f[group].attrs['first_res'] = self._opts.group_structure.first_res
        f[group].attrs['last_res'] = self._opts.group_structure.last_res

        f.close()

        # Export full xs
        self._full_xs.export_to_h5(fname)

        # Export RI table
        self._ri_table.export_to_h5(fname)


class GroupStructure(object):

    def __init__(self, name=None, group_boundaries=None, first_res=None,
                 last_res=None, fispec=None):
        if name is not None:
            if name == 'wims69e':
                self.build_wims69e()
            elif name == 'wims79':
                self.build_wims79()
            else:
                raise Exception('group structure name %s is not supported!' %
                                (name))
        else:
            if group_boundaries is None:
                raise Exception('group_boundaries should be given!')
            if first_res is None:
                raise Exception('first_res should be given!')
            if last_res is None:
                raise Exception('last_res should be given!')
            if fispec is None:
                raise Exception('fispec should be given!')
            self._group_boundaries = group_boundaries
            self._first_res = first_res
            self._last_res = last_res
            self._fispec = fispec

    def build_wims69e(self):
        self._group_boundaries = np.array([
            1.00000E+07, 6.06550E+06, 3.67900E+06, 2.23100E+06, 1.35300E+06,
            8.21000E+05, 5.00000E+05, 3.02500E+05, 1.83000E+05, 1.11000E+05,
            6.73400E+04, 4.08500E+04, 2.47800E+04, 1.50300E+04, 9.11800E+03,
            5.53000E+03, 3.51910E+03, 2.23945E+03, 1.42510E+03, 9.06899E+02,
            3.67263E+02, 1.48729E+02, 7.55014E+01, 4.80520E+01, 2.77000E+01,
            1.59680E+01, 9.87700E+00, 4.00000E+00, 3.30000E+00, 2.60000E+00,
            2.10000E+00, 1.50000E+00, 1.30000E+00, 1.15000E+00, 1.12300E+00,
            1.09700E+00, 1.07100E+00, 1.04500E+00, 1.02000E+00, 9.96000E-01,
            9.72000E-01, 9.50000E-01, 9.10000E-01, 8.50000E-01, 7.80000E-01,
            6.25000E-01, 5.00000E-01, 4.00000E-01, 3.50000E-01, 3.20000E-01,
            3.00000E-01, 2.80000E-01, 2.50000E-01, 2.20000E-01, 1.80000E-01,
            1.40000E-01, 1.00000E-01, 8.00000E-02, 6.70000E-02, 5.80000E-02,
            5.00000E-02, 4.20000E-02, 3.50000E-02, 3.00000E-02, 2.50000E-02,
            2.00000E-02, 1.50000E-02, 1.00000E-02, 5.00000E-03, 1.00000E-05])
        self._fispec = np.zeros(69)
        self._fispec[:27] = [
            2.76618619E-02, 1.16180994E-01, 2.18477324E-01, 2.32844964E-01,
            1.74191684E-01, 1.08170442E-01, 6.10514991E-02, 3.14032026E-02,
            1.55065255E-02, 7.54445791E-03, 3.62444320E-03, 1.73872744E-03,
            8.40678287E-04, 4.03833983E-04, 1.86793957E-04, 8.33578233E-05,
            4.29143147E-05, 2.21586834E-05, 1.14833065E-05, 9.10655854E-06,
            2.52294785E-06, 6.13992597E-07, 1.86799028E-07, 1.18147128E-07,
            5.79575818E-08, 2.61382667E-08, 1.37659502E-08]
        self._first_res = 12
        self._last_res = 45

    def build_wims79(self):
        self._group_boundaries = np.array([
            1.00000E+07, 6.06550E+06, 3.67900E+06, 2.23100E+06, 1.35300E+06,
            8.21000E+05, 5.00000E+05, 3.02500E+05, 1.83000E+05, 1.11000E+05,
            6.73400E+04, 4.08500E+04, 2.47800E+04, 1.50300E+04, 9.11800E+03,
            5.53000E+03, 3.51910E+03, 2.23945E+03, 1.42510E+03, 9.06899E+02,
            750.0,       500.0,       3.67263E+02, 325.0,       225.0,
            200.0,       1.48729E+02, 140.0,       130.0,       110.0,
            90.0,        85.0,        7.55014E+01, 4.80520E+01, 2.77000E+01,
            1.59680E+01, 9.87700E+00, 4.00000E+00, 3.30000E+00, 2.60000E+00,
            2.10000E+00, 1.50000E+00, 1.30000E+00, 1.15000E+00, 1.12300E+00,
            1.09700E+00, 1.07100E+00, 1.04500E+00, 1.02000E+00, 9.96000E-01,
            9.72000E-01, 9.50000E-01, 9.10000E-01, 8.50000E-01, 7.80000E-01,
            6.25000E-01, 5.00000E-01, 4.00000E-01, 3.50000E-01, 3.20000E-01,
            3.00000E-01, 2.80000E-01, 2.50000E-01, 2.20000E-01, 1.80000E-01,
            1.40000E-01, 1.00000E-01, 8.00000E-02, 6.70000E-02, 5.80000E-02,
            5.00000E-02, 4.20000E-02, 3.50000E-02, 3.00000E-02, 2.50000E-02,
            2.00000E-02, 1.50000E-02, 1.00000E-02, 5.00000E-03, 1.00000E-05])
        self._fispec = np.zeros(79)
        self._first_res = 12
        self._last_res = 55

    def is_same_as(self, another):
        if not isinstance(another, GroupStructure):
            return False
        if self._first_res != another._first_res:
            return False
        if self._last_res != another._last_res:
            return False
        for x, y in zip(self._fispec, another._fispec):
            if x != y:
                return False
        for x, y in zip(self._group_boundaries, another._group_boundaries):
            if x != y:
                return False
        return True

    @property
    def res_group_bnds(self):
        return self._group_boundaries[self._first_res:self._last_res+1]

    @property
    def group_boundaries(self):
        return self._group_boundaries

    @property
    def fispec(self):
        return self._fispec

    @property
    def first_res(self):
        return self._first_res

    @property
    def last_res(self):
        return self._last_res

    @property
    def ng(self):
        return len(self._group_boundaries) - 1

    @property
    def n_res(self):
        return self._last_res - self._first_res


def export_homo_problem_xml(nuclide, dilution, temperature,
                            background_nuclide, fission_nuclide=None,
                            fisnuc_refdil=None):
    # Material is composed of background H-1 and the object nuclide
    mat = openmc.Material(material_id=1, name='mat')
    mat.set_density('atom/b-cm', 0.069335)
    mat.add_nuclide(nuclide, 1.0)
    if nuclide != background_nuclide:
        mat.add_nuclide(background_nuclide, dilution / 20.478001)
    if fission_nuclide is not None:
        if fission_nuclide != nuclide:
            if fisnuc_refdil is None:
                raise Exception(
                    'fisnuc_refdil should be given')
            fisnuc = openmc.Nuclide(fission_nuclide)
            mat.add_nuclide(fisnuc, dilution / fisnuc_refdil)
    materials_file = openmc.Materials([mat])
    materials_file.export_to_xml()

    # Cell is box with reflective boundary
    x1 = openmc.XPlane(surface_id=1, x0=-1)
    x2 = openmc.XPlane(surface_id=2, x0=1)
    y1 = openmc.YPlane(surface_id=3, y0=-1)
    y2 = openmc.YPlane(surface_id=4, y0=1)
    z1 = openmc.ZPlane(surface_id=5, z0=-1)
    z2 = openmc.ZPlane(surface_id=6, z0=1)
    for surface in [x1, x2, y1, y2, z1, z2]:
        surface.boundary_type = 'reflective'
    box = openmc.Cell(cell_id=1, name='box')
    box_region = +x1 & -x2 & +y1 & -y2 & +z1 & -z2
    box.region = box_region
    box.fill = mat
    root = openmc.Universe(universe_id=0, name='root universe')
    root.add_cell(box)
    geometry = openmc.Geometry(root)
    geometry.export_to_xml()

    return mat, geometry


class FullXs(object):

    def __init__(self, opts):
        if opts.nuclide is None:
            raise Exception('nuclide of opts should not be None')
        self._nuclide = opts.nuclide

        # Set default settings
        self._temperatures = opts.temperatures
        self._legendre_order = opts.legendre_order
        self._group_structure = opts.group_structure
        self._reference_dilution = opts.reference_dilution
        self._slowdown_nuclide = opts.slowdown_nuclide
        self._fission_nuclide = opts.fission_nuclide
        self._fisnuc_refdil = opts.fisnuc_refdil
        self._batches = opts.batches
        self._particles = opts.particles
        self._inactive = opts.inactive

    def _export_fs_xml(self, temperature):
        # Export geometry and materials of homogeneous problem
        self._material, self._geometry = export_homo_problem_xml(
            self._nuclide, self._reference_dilution, temperature,
            self._slowdown_nuclide)

        # Calculate number density of object nuclide
        sum_dens = 0.0
        nuclides = self._material.get_nuclide_densities()
        for nuc in nuclides:
            sum_dens += nuclides[nuc][1]
        self._nuclide_density \
            = self._material._density * nuclides[self._nuclide][1] / sum_dens

        # Set the running parameters
        settings_file = openmc.Settings()
        settings_file.run_mode = 'fixed source'
        settings_file.batches = self._batches
        settings_file.particles = self._particles
        settings_file.no_nu = True
        bounds = [-1, -1, -1, 1, 1, 1]
        uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:],
                                        only_fissionable=False)
        watt_dist = openmc.stats.Watt()
        settings_file.source = openmc.source.Source(space=uniform_dist,
                                                    energy=watt_dist)
        settings_file.temperature = {'default': temperature}
        settings_file.create_fission_neutrons = False
        settings_file.cutoff \
            = {'energy': self._group_structure.group_boundaries[-1]}
        settings_file.export_to_xml()

        # Create tallies
        tallies = openmc.Tallies()

        grp_bnds = [val for val in
                    sorted(self._group_structure.group_boundaries)]
        energy_filter = openmc.EnergyFilter(grp_bnds)
        energy_out_filter = openmc.EnergyoutFilter(grp_bnds)
        cell_filter = openmc.CellFilter((1, ))

        for score in ['total', 'fission', 'nu-fission', 'absorption']:
            energy_tally = openmc.Tally()
            energy_tally.estimator = 'tracklength'
            energy_tally.filters = [energy_filter, cell_filter]
            energy_tally.nuclides = [self._nuclide]
            energy_tally.scores = [score]
            tallies.append(energy_tally)

        for i in range(self._legendre_order + 1):
            energy_out_tally = openmc.Tally()
            energy_out_tally.estimator = 'analog'
            energy_out_tally.filters = [energy_filter, energy_out_filter,
                                        cell_filter]
            energy_out_tally.nuclides = [self._nuclide]
            energy_out_tally.scores = ['nu-scatter-%s' % (i)]
            tallies.append(energy_out_tally)

        flux_tally = openmc.Tally()
        flux_tally.estimator = 'tracklength'
        flux_tally.filters = [energy_filter, cell_filter]
        flux_tally.scores = ['flux']
        tallies.append(flux_tally)

        tallies.export_to_xml()

    def _export_eig_xml(self):
        # Export geometry and materials of homogeneous problem
        export_homo_problem_xml(self._nuclide, self._reference_dilution,
                                self._temperatures[0],
                                self._slowdown_nuclide,
                                fission_nuclide=self._fission_nuclide,
                                fisnuc_refdil=self._fisnuc_refdil)

        # Set the running parameters
        settings_file = openmc.Settings()
        settings_file.run_mode = 'eigenvalue'
        settings_file.batches = self._batches
        settings_file.inactive = self._inactive
        settings_file.particles = self._particles
        bounds = [-1, -1, -1, 1, 1, 1]
        uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:],
                                        only_fissionable=False)
        settings_file.source = openmc.source.Source(space=uniform_dist)
        settings_file.temperature = {'default': self._temperatures[0]}
        settings_file.export_to_xml()

        # Create tallies
        energy_out_tally = openmc.Tally()
        grp_bnds = [val for val in
                    sorted(self._group_structure.group_boundaries)]
        energy_filter = openmc.EnergyFilter(grp_bnds)
        energy_out_filter = openmc.EnergyoutFilter(grp_bnds)
        cell_filter = openmc.CellFilter((1, ))
        energy_out_tally.estimator = 'analog'
        energy_out_tally.filters = [energy_filter, energy_out_filter,
                                    cell_filter]
        energy_out_tally.nuclides = [self._nuclide]
        energy_out_tally.scores = ['nu-fission']

        tallies = openmc.Tallies()
        tallies.append(energy_out_tally)
        tallies.export_to_xml()

    def build_library(self):
        print('processing %s/full_xs ...' % (self._nuclide))

        # Initialize multi-group cross sections (full xs part)
        ng = self._group_structure.ng
        ng_in = ng
        ng_out = ng
        n_temp = len(self._temperatures)
        self._fissionable = False
        self._total = np.zeros((n_temp, ng))
        self._fission = np.zeros((n_temp, ng))
        self._nu_fission = np.zeros((n_temp, ng))
        self._nu_fission_matrix = np.zeros((ng_in, ng_out))
        self._flux_fix = np.zeros((n_temp, ng))
        self._nu_scatter = np.zeros((n_temp, ng_in, ng_out,
                                     self._legendre_order + 1))
        self._chi = np.zeros(ng)

        for itemp, temperature in enumerate(self._temperatures):
            print('processing %s/full_xs/temp%i ...' %
                  (self._nuclide, itemp))

            # Export fixed source input files
            self._export_fs_xml(temperature)

            # Run OpenMC
            _run_openmc()

            # Load the tally data from statepoint
            statepoint = glob(os.path.join(
                os.getcwd(), "statepoint.%s.*" % (self._batches)))[0]
            self._load_fix_statepoint(statepoint, itemp)

        if self._fissionable:
            print('processing %s/full_xs/chi ...' % (self._nuclide))
            # Process chi for fissionable nuclide
            # Export eigenvalue input files
            self._export_eig_xml()

            # Run OpenMC
            _run_openmc()

            # Load the tally data from statepoint
            statepoint = glob(os.path.join(
                os.getcwd(), "statepoint.%s.*" % (self._batches)))[0]
            self._load_eig_statepoint(statepoint)

    def export_to_h5(self, fname):
        f = h5py.File(fname)
        root_group = '/' + self._nuclide
        ng = self._group_structure.ng
        n_temp = len(self._temperatures)

        # Remove existing full xs data
        group = root_group + '/full_xs'
        if group in f:
            del f[group]
        f.create_group(group)

        # Whether fissionable
        if self._fissionable:
            f[group].attrs['fissionable'] = 1
        else:
            f[group].attrs['fissionable'] = 0

        # Legendre order
        f[group].attrs['legendre_order'] = self._legendre_order

        # Temperatures
        f[group + '/temperatures'] = self._temperatures

        for itemp in range(n_temp):
            group = '%s/full_xs/temp%s' % (
                root_group, itemp)

            # Total cross sections
            f[group + '/total'] = self._total[itemp, :]

            # Scattering matrix
            for il in range(self._legendre_order + 1):
                for ig_to in range(ng):
                    ig0, ig1, scatter \
                        = _condense_scatter(
                            self._nu_scatter[itemp, :, ig_to, il],
                            ig_to)
                    dset = group + '/scatter/lo%s/to%s' % (il, ig_to)
                    f[dset] = scatter
                    f[dset].attrs['ig0'] = ig0
                    f[dset].attrs['ig1'] = ig1

            if self._fissionable:
                # Fission chi
                f[group + '/chi'] = self._chi

                # Fission cross sections
                f[group + '/fission'] = self._fission[itemp, :]

                # Nu fission cross sections
                f[group + '/nu_fission'] = self._nu_fission[itemp, :]

        f.close()

    def _load_fix_statepoint(self, statepoint, itemp):
        sp = openmc.StatePoint(statepoint)
        ng = self._group_structure.ng
        first_res = self._group_structure.first_res

        # Get flux
        self._flux_fix[itemp, :] \
            = sp.get_tally(scores=['flux']).sum[:, 0, 0][::-1] \
            * self._nuclide_density

        # Get total xs
        self._total[itemp, :] \
            = sp.get_tally(scores=['total'], nuclides=[self._nuclide])\
                .sum[:, 0, 0][::-1]
        self._total[itemp, :] /= self._flux_fix[itemp, :]

        # Get fission xs
        self._fission[itemp, :] \
            = sp.get_tally(scores=['fission'], nuclides=[self._nuclide])\
                .sum[:, 0, 0][::-1]
        self._fission[itemp, :] /= self._flux_fix[itemp, :]

        # Get nu fission xs
        self._nu_fission[itemp, :] \
            = sp.get_tally(scores=['nu-fission'], nuclides=[self._nuclide])\
                .sum[:, 0, 0][::-1]
        self._nu_fission[itemp, :] /= self._flux_fix[itemp, :]

        # Get absorption xs
        absorb = sp.get_tally(scores=['absorption'], nuclides=[self._nuclide])\
                   .sum[:, 0, 0][::-1]
        absorb[:] /= self._flux_fix[itemp, :]

        # Get scattering matrix (analog)
        for i in range(self._legendre_order + 1):
            self._nu_scatter[itemp, :, :, i] \
                = sp.get_tally(scores=['nu-scatter-%s' % i],
                               nuclides=[self._nuclide])\
                    .sum[:, 0, 0][::-1].reshape(ng, ng)
            for ig in range(ng):
                self._nu_scatter[itemp, ig, :, i] /= self._flux_fix[itemp, ig]

        # Compute nu scatter from total and absorption in the resonance and
        # thermal energy ranges. As (n,xn) reactions are threshold reactions,
        # nu_scatter is scatter in the resonance and thermal energy ranges.
        nu_scatter_tl = self._total[itemp, :] - absorb[:]
        nu_scatter_al = np.sum(self._nu_scatter[itemp, :, :, 0], 1)
        for ig in range(first_res, ng):
            self._nu_scatter[itemp, ig, :, 0] \
                *= nu_scatter_tl[ig] / nu_scatter_al[ig]

        # Determine whether is fissionable nuclide
        if sum(self._fission[itemp, :]) != 0.0:
            self._fissionable = True

    def _load_eig_statepoint(self, statepoint):
        sp = openmc.StatePoint(statepoint)
        ng = self._group_structure.ng

        # Get fission matrix
        self._nu_fission_matrix[:, :] \
            = sp.get_tally(scores=['nu-fission'], nuclides=[self._nuclide])\
                .sum[:, 0, 0][::-1].reshape(ng, ng)

        # Calculate fission chi
        self._chi[:] = self._nu_fission_matrix.sum(axis=0) \
            / self._nu_fission_matrix.sum()


class RItable(object):

    def __init__(self, opts):
        if opts.nuclide is None:
            raise Exception('nuclide of opts should not be None')
        self._nuclide = opts.nuclide

        # Set default settings
        self._dilutions = opts.dilutions
        self._temperatures = opts.temperatures
        self._group_structure = opts.group_structure
        self._background_nuclide = opts.background_nuclide
        self._batches = opts.batches
        self._particles = opts.particles
        self._resfis_method = opts.resfis_method
        self._has_res = opts.has_res
        self._has_resfis = opts.has_resfis
        self._ri_use_openmc = opts.ri_use_openmc
        self._find_nearest_temp = opts.find_nearest_temp
        self._nu = opts.nu

        if not self._ri_use_openmc and self._has_resfis and self._nu is None:
            raise Exception('nu should be given when not use OpenMC')

    def _export_xml(self, temperature, dilution):
        # Export geometry and materials of homogeneous problem
        self._material, self._geometry = export_homo_problem_xml(
            self._nuclide, dilution, temperature, self._background_nuclide)

        # Calculate number density of object nuclide
        sum_dens = 0.0
        nuclides = self._material.get_nuclide_densities()
        for nuc in nuclides:
            sum_dens += nuclides[nuc][1]
        self._nuclide_density \
            = self._material._density * nuclides[self._nuclide][1] / sum_dens

        # Set the running parameters
        settings_file = openmc.Settings()
        settings_file.run_mode = 'fixed source'
        settings_file.batches = self._batches
        settings_file.particles = self._particles
        settings_file.create_fission_neutrons = False
        settings_file.cutoff \
            = {'energy': self._group_structure.res_group_bnds[-1]}
        bounds = [-1, -1, -1, 1, 1, 1]
        uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:],
                                        only_fissionable=False)
        watt_dist = openmc.stats.Watt()
        settings_file.source = openmc.source.Source(space=uniform_dist,
                                                    energy=watt_dist)
        settings_file.temperature = {'default': temperature}
        settings_file.export_to_xml()

        # Create tallies
        tallies = openmc.Tallies()

        grp_bnds = [val for val in
                    sorted(self._group_structure.res_group_bnds)]
        energy_filter = openmc.EnergyFilter(grp_bnds)
        cell_filter = openmc.CellFilter((1, ))

        flux_tally = openmc.Tally()
        flux_tally.estimator = 'tracklength'
        flux_tally.filters = [energy_filter, cell_filter]
        flux_tally.scores = ['flux']
        tallies.append(flux_tally)

        scores = ['absorption', 'scatter']
        if self._has_resfis:
            scores.append('nu-fission')
        for score in scores:
            reaction_tally = openmc.Tally()
            reaction_tally.estimator = 'tracklength'
            reaction_tally.filters = [energy_filter, cell_filter]
            reaction_tally.nuclides = [self._nuclide]
            reaction_tally.scores = [score]
            tallies.append(reaction_tally)

        tallies.export_to_xml()

    def _obtain_sd_xs(self, temperature, itemp):
        global _cross_sections

        # Initialize slowing down solver
        emax = self._group_structure.res_group_bnds[0]
        emin = self._group_structure.res_group_bnds[-1]
        sd = SDSolver()
        sd.setErgGrpBnd(self._group_structure.res_group_bnds)
        sd.setNumNuclide(2)
        sd.setSolErgBnd(emin, emax)

        # Initialize resonant nuclides of slowing down solver
        celib = LibraryCe(_cross_sections)
        hflib = celib.get_nuclide(self._nuclide, temperature, emax, emin,
                                  self._has_resfis,
                                  find_nearest_temp=self._find_nearest_temp)
        resnuc = sd.getNuclide(0)
        resnuc.setName(self._nuclide)
        resnuc.setHFLibrary(hflib)
        resnuc.setNumDens(np.ones(len(self._dilutions)))

        # Initialize background nuclide of slowing down solver
        baknuc = sd.getNuclide(1)
        baknuc.setName('H1')
        baknuc.setAwr(0.9991673)
        baknuc.setPotential(1.0)
        baknuc.setNumDens(np.array(self._dilutions))

        # Solve slowing down equation
        sd.computeFlux()
        sd.computeMgXs()

        for ig in range(self._group_structure.n_res):
            for idlt in range(len(self._dilutions)):
                self._flux[itemp, ig, idlt] = sd.getMgFlux(ig, idlt)
                xs_tot = resnuc.getMgTotal(ig, idlt)
                self._nu_scatter[itemp, ig, idlt] \
                    = resnuc.getMgScatter(ig, idlt)
                self._absorption[itemp, ig, idlt] \
                    = xs_tot - self._nu_scatter[itemp, ig, idlt]
                if self._has_resfis:
                    self._nu_fission[itemp, ig, idlt] \
                        = self._nu[ig] * resnuc.getMgFission(ig, idlt)

        del sd

    def _obtain_openmc_xs(self, temperature, dilution, itemp, idlt):
        # Export input files
        self._export_xml(temperature, dilution)

        # Run OpenMC
        _run_openmc()

        # Load the tally data from statepoint
        statepoint = glob(os.path.join(
            os.getcwd(), "statepoint.%s.*" % (self._batches)))[0]
        self._load_statepoint(statepoint, itemp, idlt)

    def build_library(self):
        global _cross_sections

        print('processing %s/resonance ...' % self._nuclide)

        # Get A, Z and atomic weight ratio
        self._A, self._Z, self._awr \
            = _get_A_Z_awr(_cross_sections, self._nuclide)

        # Don't build library if no resonance
        if not self._has_res:
            return

        # Whether has resonance fission
        if self._resfis_method == RESONANCE_FISSION_AUTO:
            self._has_resfis = _has_resfis(self._A, self._Z)

        # Initialize multi-group cross sections (resonance xs table part)
        n_res = self._group_structure.n_res
        n_temp = len(self._temperatures)
        n_dilution = len(self._dilutions)
        self._flux = np.zeros((n_temp, n_res, n_dilution))
        self._absorption = np.zeros((n_temp, n_res, n_dilution))
        self._nu_scatter = np.zeros((n_temp, n_res, n_dilution))
        if self._has_resfis:
            self._nu_fission = np.zeros((n_temp, n_res, n_dilution))

        for itemp, temperature, in enumerate(self._temperatures):
            print('processing %s/resonance/temp%i ...' %
                  (self._nuclide, itemp))
            if self._ri_use_openmc:
                for idlt, dilution in enumerate(self._dilutions):
                    print('processing %s/resonance/temp%i/dlt%i ...' %
                          (self._nuclide, itemp, idlt))
                    self._obtain_openmc_xs(temperature, dilution, itemp, idlt)
            else:
                self._obtain_sd_xs(temperature, itemp)

    def _load_statepoint(self, statepoint, itemp, idlt):
        sp = openmc.StatePoint(statepoint)

        # Get flux
        self._flux[itemp, :, idlt] \
            = sp.get_tally(scores=['flux']).sum[:, 0, 0][::-1] * \
            self._nuclide_density

        # Get absorption xs
        self._absorption[itemp, :, idlt] \
            = sp.get_tally(scores=['absorption'], nuclides=[self._nuclide])\
                .sum[:, 0, 0][::-1]
        self._absorption[itemp, :, idlt] /= self._flux[itemp, :, idlt]

        # Get scatter xs
        self._nu_scatter[itemp, :, idlt] \
            = sp.get_tally(scores=['scatter'], nuclides=[self._nuclide])\
                .sum[:, 0, 0][::-1]
        self._nu_scatter[itemp, :, idlt] /= self._flux[itemp, :, idlt]

        # Get nu fission xs
        if self._has_resfis:
            self._nu_fission[itemp, :, idlt] \
                = sp.get_tally(scores=['nu-fission'],
                               nuclides=[self._nuclide]).sum[:, 0, 0][::-1]
            self._nu_fission[itemp, :, idlt] \
                /= self._flux[itemp, :, idlt]

    def export_to_h5(self, fname):
        f = h5py.File(fname)
        root_group = '/' + self._nuclide

        # Create resonance group
        group = root_group + '/resonance'
        if group in f:
            del f[group]
        f.create_group(group)

        # Whether has resonance
        if self._has_res:
            f[group].attrs['has_res'] = 1
        else:
            f[group].attrs['has_res'] = 0

        # Whether has resonance fission
        if self._has_resfis:
            f[group].attrs['has_resfis'] = 1
        else:
            f[group].attrs['has_resfis'] = 0

        # Average potential
        f[group + '/average_potential'] = average_potentials(self._nuclide)

        # Average lambda
        f[group + '/average_lambda'] = average_lambda(self._A)

        if self._has_res:
            # Temperatures
            f[group + '/temperatures'] = self._temperatures

            # Dilution cross sections
            f[group + '/dilutions'] = self._dilutions

            # Resonance cross sections
            for itemp in range(len(self._temperatures)):
                f['{0}/temp{1}/absorption'.format(group, itemp)]\
                    = self._absorption[itemp, ...]
                f['{0}/temp{1}/scatter'.format(group, itemp)]\
                    = self._nu_scatter[itemp, ...]
                if self._has_resfis:
                    f['{0}/temp{1}/nu_fission'.format(group, itemp)]\
                        = self._nu_fission[itemp, ...]

        f.close()

if __name__ == '__main__':
    libname = 'wims79'
    lib_fname = 'jeff-3.2-%s.h5' % libname
    set_default_settings(batches=30, inactive=10, particles=1000)
    group_structure = GroupStructure(libname)
    opts_list = []

    # Options for generating U238
    opts_u238 = MicroMgXsOptions()
    opts_u238.nuclide = 'U238'
    opts_u238.has_res = True
    opts_u238.reference_dilution = 45.0
    # opts_u238.temperatures = [293.6]
    # opts_u238.dilutions = [28.0]
    opts_u238.ri_use_openmc = False
    opts_u238.group_structure = group_structure
    opts_list.append(opts_u238)

    # # Options for generating U235
    # opts_u235 = MicroMgXsOptions()
    # opts_u235.nuclide = 'U235'
    # opts_u235.has_res = True
    # opts_u235.reference_dilution = 800.0
    # opts_list.append(opts_u235)

    # Options for generating H1
    opts_h1 = MicroMgXsOptions()
    opts_h1.nuclide = 'H1'
    opts_h1.group_structure = group_structure
    opts_list.append(opts_h1)

    # # Options for generating O16
    # opts_o16 = MicroMgXsOptions()
    # opts_o16.nuclide = 'O16'
    # opts_list.append(opts_o16)

    lib = MicroMgXsLibrary(opts_list, lib_fname)
    lib.build_library_h5()
