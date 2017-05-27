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

RESONANCE_FISSION_USER = 2
DEFAULT_BATCHES = 1000
DEFAULT_INACTIVE = 100
DEFAULT_PARTICLES = 100000
DEFAULT_RI_BATCHES = 100
DEFAULT_RI_PARTICLES = 1000
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


def set_default_settings(batches=None, inactive=None, particles=None,
                         ri_batches=None, ri_particles=None):
    global DEFAULT_BATCHES, DEFAULT_INACTIVE, DEFAULT_PARTICLES,\
        DEFAULT_RI_BATCHES, DEFAULT_RI_PARTICLES
    if batches is not None:
        DEFAULT_BATCHES = batches
    if inactive is not None:
        DEFAULT_INACTIVE = inactive
    if particles is not None:
        DEFAULT_PARTICLES = particles
    if ri_batches is not None:
        DEFAULT_RI_BATCHES = ri_batches
    if ri_particles is not None:
        DEFAULT_RI_PARTICLES = ri_particles


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
        self._ri_batches = DEFAULT_RI_BATCHES
        self._ri_particles = DEFAULT_RI_PARTICLES
        self._has_res = False
        self._has_resfis = False
        self._ri_use_openmc = False
        self._find_nearest_temp = False
        self._nu = None

    @property
    def nuclide(self):
        return self._nuclide

    @nuclide.setter
    def nuclide(self, nuclide):
        self._nuclide = nuclide

    @property
    def ri_batches(self):
        return self._ri_batches

    @ri_batches.setter
    def ri_batches(self, ri_batches):
        self._ri_batches = ri_batches

    @property
    def ri_particles(self):
        return self._ri_particles

    @ri_particles.setter
    def ri_particles(self, ri_particles):
        self._ri_particles = ri_particles

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
            elif name == 'wims172':
                self.build_wims172()
            elif name == 'shem281':
                self.build_shem281()
            elif name == 'shem361':
                self.build_shem361()
            elif name == 'helios45':
                self.build_helios45()
            elif name == 'helios190':
                self.build_helios190()
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
            2.00000E+07, 6.06550E+06, 3.67900E+06, 2.23100E+06, 1.35300E+06,
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
            2.00000E-02, 1.50000E-02, 1.00000E-02, 5.00000E-03, 1.00000E-05
        ])
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

    def build_wims172(self):
        self._group_boundaries = np.array([
            1.96403E+07, 1.73325E+07, 1.49182E+07, 1.38403E+07, 1.16183E+07,
            1.00000E+07, 8.18731E+06, 6.70320E+06, 6.06531E+06, 5.48812E+06,
            4.49329E+06, 3.67879E+06, 3.01194E+06, 2.46597E+06, 2.23130E+06,
            2.01897E+06, 1.65299E+06, 1.35335E+06, 1.22456E+06, 1.10803E+06,
            1.00259E+06, 9.07180E+05, 8.20850E+05, 6.08101E+05, 5.50232E+05,
            4.97871E+05, 4.50492E+05, 4.07622E+05, 3.01974E+05, 2.73237E+05,
            2.47235E+05, 1.83156E+05, 1.22773E+05, 1.11090E+05, 8.22975E+04,
            6.73795E+04, 5.51656E+04, 4.08677E+04, 3.69786E+04, 2.92830E+04,
            2.73944E+04, 2.47875E+04, 1.66156E+04, 1.50344E+04, 1.11378E+04,
            9.11882E+03, 7.46586E+03, 5.53085E+03, 5.00450E+03, 3.52662E+03,
            3.35463E+03, 2.24867E+03, 2.03468E+03, 1.50733E+03, 1.43382E+03,
            1.23410E+03, 1.01039E+03, 9.14242E+02, 7.48518E+02, 6.77287E+02,
            4.53999E+02, 3.71703E+02, 3.04325E+02, 2.03995E+02, 1.48625E+02,
            1.36742E+02, 9.16609E+01, 7.56736E+01, 6.79040E+01, 5.55951E+01,
            5.15780E+01, 4.82516E+01, 4.55174E+01, 4.01690E+01, 3.72665E+01,
            3.37201E+01, 3.05113E+01, 2.76077E+01, 2.49805E+01, 2.26033E+01,
            1.94548E+01, 1.59283E+01, 1.37096E+01, 1.12245E+01, 9.90555E+00,
            9.18981E+00, 8.31529E+00, 7.52398E+00, 6.16012E+00, 5.34643E+00,
            5.04348E+00, 4.12925E+00, 4.00000E+00, 3.38075E+00, 3.30000E+00,
            2.76792E+00, 2.72000E+00, 2.60000E+00, 2.55000E+00, 2.36000E+00,
            2.13000E+00, 2.10000E+00, 2.02000E+00, 1.93000E+00, 1.84000E+00,
            1.75500E+00, 1.67000E+00, 1.59000E+00, 1.50000E+00, 1.47500E+00,
            1.44498E+00, 1.37000E+00, 1.33750E+00, 1.30000E+00, 1.23500E+00,
            1.17000E+00, 1.15000E+00, 1.12535E+00, 1.11000E+00, 1.09700E+00,
            1.07100E+00, 1.04500E+00, 1.03500E+00, 1.02000E+00, 9.96000E-01,
            9.86000E-01, 9.72000E-01, 9.50000E-01, 9.30000E-01, 9.10000E-01,
            8.60000E-01, 8.50000E-01, 7.90000E-01, 7.80000E-01, 7.05000E-01,
            6.25000E-01, 5.40000E-01, 5.00000E-01, 4.85000E-01, 4.33000E-01,
            4.00000E-01, 3.91000E-01, 3.50000E-01, 3.20000E-01, 3.14500E-01,
            3.00000E-01, 2.80000E-01, 2.48000E-01, 2.20000E-01, 1.89000E-01,
            1.80000E-01, 1.60000E-01, 1.40000E-01, 1.34000E-01, 1.15000E-01,
            1.00000E-01, 9.50000E-02, 8.00000E-02, 7.70000E-02, 6.70000E-02,
            5.80000E-02, 5.00000E-02, 4.20000E-02, 3.50000E-02, 3.00000E-02,
            2.50000E-02, 2.00000E-02, 1.50000E-02, 1.00000E-02, 6.90000E-03,
            5.00000E-03, 3.00000E-03, 1.00000E-05
        ])
        self._fispec = np.zeros(172)
        self._first_res = 45
        self._last_res = 92

    def build_shem281(self):
        self._group_boundaries = np.array([
            1.964E+07, 1.492E+07, 1.384E+07, 1.162E+07, 1.000E+07, 9.048E+06,
            8.187E+06, 7.408E+06, 6.703E+06, 6.065E+06, 4.966E+06, 4.066E+06,
            3.329E+06, 2.725E+06, 2.231E+06, 1.901E+06, 1.637E+06, 1.406E+06,
            1.337E+06, 1.287E+06, 1.162E+06, 1.051E+06, 9.511E+05, 8.600E+05,
            7.065E+05, 5.784E+05, 4.940E+05, 4.560E+05, 4.125E+05, 3.839E+05,
            3.206E+05, 2.678E+05, 2.300E+05, 1.950E+05, 1.650E+05, 1.400E+05,
            1.228E+05, 1.156E+05, 9.466E+04, 8.230E+04, 6.738E+04, 5.517E+04,
            4.992E+04, 4.087E+04, 3.698E+04, 3.346E+04, 2.928E+04, 2.739E+04,
            2.610E+04, 2.500E+04, 2.270E+04, 1.858E+04, 1.620E+04, 1.490E+04,
            1.360E+04, 1.114E+04, 9.119E+03, 7.466E+03, 6.113E+03, 5.005E+03,
            4.097E+03, 3.481E+03, 2.996E+03, 2.579E+03, 2.220E+03, 1.910E+03,
            1.614E+03, 1.345E+03, 1.135E+03, 1.065E+03, 9.075E+02, 7.485E+02,
            6.128E+02, 5.017E+02, 4.108E+02, 3.536E+02, 3.199E+02, 2.838E+02,
            2.418E+02, 1.980E+02, 1.621E+02, 1.327E+02, 1.086E+02, 8.895E+01,
            7.505E+01, 6.144E+01, 5.267E+01, 4.579E+01, 4.400E+01, 4.017E+01,
            3.372E+01, 2.761E+01, 2.459E+01, 2.252E+01, 2.238E+01, 2.216E+01,
            2.200E+01, 2.170E+01, 2.149E+01, 2.134E+01, 2.123E+01, 2.114E+01,
            2.106E+01, 2.098E+01, 2.077E+01, 2.068E+01, 2.060E+01, 2.052E+01,
            2.042E+01, 2.028E+01, 2.007E+01, 1.960E+01, 1.939E+01, 1.920E+01,
            1.908E+01, 1.796E+01, 1.776E+01, 1.756E+01, 1.745E+01, 1.683E+01,
            1.655E+01, 1.605E+01, 1.578E+01, 1.487E+01, 1.473E+01, 1.460E+01,
            1.447E+01, 1.425E+01, 1.405E+01, 1.355E+01, 1.333E+01, 1.260E+01,
            1.247E+01, 1.231E+01, 1.213E+01, 1.198E+01, 1.182E+01, 1.171E+01,
            1.159E+01, 1.127E+01, 1.105E+01, 1.080E+01, 1.058E+01, 9.500E+00,
            9.140E+00, 8.980E+00, 8.800E+00, 8.674E+00, 8.524E+00, 8.300E+00,
            8.130E+00, 7.970E+00, 7.840E+00, 7.740E+00, 7.600E+00, 7.380E+00,
            7.140E+00, 6.994E+00, 6.918E+00, 6.870E+00, 6.835E+00, 6.811E+00,
            6.792E+00, 6.776E+00, 6.760E+00, 6.742E+00, 6.717E+00, 6.631E+00,
            6.606E+00, 6.588E+00, 6.572E+00, 6.556E+00, 6.539E+00, 6.515E+00,
            6.482E+00, 6.432E+00, 6.360E+00, 6.280E+00, 6.160E+00, 6.060E+00,
            5.960E+00, 5.800E+00, 5.720E+00, 5.620E+00, 5.530E+00, 5.488E+00,
            5.410E+00, 5.380E+00, 5.320E+00, 5.210E+00, 5.110E+00, 4.933E+00,
            4.768E+00, 4.420E+00, 4.310E+00, 4.220E+00, 4.000E+00, 3.882E+00,
            3.712E+00, 3.543E+00, 3.142E+00, 2.884E+00, 2.775E+00, 2.741E+00,
            2.720E+00, 2.700E+00, 2.640E+00, 2.620E+00, 2.590E+00, 2.550E+00,
            2.470E+00, 2.330E+00, 2.273E+00, 2.217E+00, 2.157E+00, 2.070E+00,
            1.990E+00, 1.900E+00, 1.780E+00, 1.669E+00, 1.588E+00, 1.520E+00,
            1.444E+00, 1.410E+00, 1.381E+00, 1.331E+00, 1.293E+00, 1.251E+00,
            1.214E+00, 1.170E+00, 1.148E+00, 1.130E+00, 1.116E+00, 1.104E+00,
            1.092E+00, 1.078E+00, 1.035E+00, 1.021E+00, 1.009E+00, 9.965E-01,
            9.820E-01, 9.640E-01, 9.440E-01, 9.200E-01, 8.800E-01, 8.200E-01,
            7.200E-01, 6.250E-01, 5.950E-01, 5.550E-01, 5.200E-01, 4.750E-01,
            4.316E-01, 3.900E-01, 3.530E-01, 3.250E-01, 3.050E-01, 2.800E-01,
            2.550E-01, 2.312E-01, 2.096E-01, 1.900E-01, 1.619E-01, 1.380E-01,
            1.200E-01, 1.043E-01, 8.980E-02, 7.650E-02, 6.520E-02, 5.550E-02,
            4.730E-02, 4.030E-02, 3.440E-02, 2.930E-02, 2.494E-02, 2.001E-02,
            1.483E-02, 1.045E-02, 7.145E-03, 4.556E-03, 2.500E-03, 1.000E-05
        ])
        self._fispec = np.zeros(281)
        self._first_res = 56
        self._last_res = 93

    def build_shem361(self):
        self._group_boundaries = np.array([
            1.964E+07, 1.492E+07, 1.384E+07, 1.162E+07, 1.000E+07, 9.048E+06,
            8.187E+06, 7.408E+06, 6.703E+06, 6.065E+06, 4.966E+06, 4.066E+06,
            3.329E+06, 2.725E+06, 2.231E+06, 1.901E+06, 1.637E+06, 1.406E+06,
            1.337E+06, 1.287E+06, 1.162E+06, 1.051E+06, 9.511E+05, 8.600E+05,
            7.065E+05, 5.784E+05, 4.940E+05, 4.560E+05, 4.125E+05, 3.839E+05,
            3.206E+05, 2.678E+05, 2.300E+05, 1.950E+05, 1.650E+05, 1.400E+05,
            1.228E+05, 1.156E+05, 9.466E+04, 8.230E+04, 6.738E+04, 5.517E+04,
            4.992E+04, 4.087E+04, 3.698E+04, 3.346E+04, 2.928E+04, 2.739E+04,
            2.610E+04, 2.500E+04, 2.270E+04, 1.858E+04, 1.620E+04, 1.490E+04,
            1.360E+04, 1.1138E+04, 9.1188E+03, 7.4658E+03, 6.1125E+03,
            5.0045E+03, 4.0973E+03, 3.4811E+03, 2.9962E+03, 2.7002E+03,
            2.3973E+03, 2.0841E+03, 1.8118E+03, 1.5862E+03, 1.3436E+03,
            1.1347E+03, 1.0643E+03, 9.8249E+02, 9.0968E+02, 8.3222E+02,
            7.4852E+02, 6.7729E+02, 6.4684E+02, 6.1283E+02, 6.0010E+02,
            5.9294E+02, 5.7715E+02, 5.3920E+02, 5.0175E+02, 4.5400E+02,
            4.1909E+02, 3.9076E+02, 3.7170E+02, 3.5357E+02, 3.3532E+02,
            3.1993E+02, 2.9592E+02, 2.8833E+02, 2.8489E+02, 2.7647E+02,
            2.6830E+02, 2.5675E+02, 2.4180E+02, 2.3559E+02, 2.2432E+02,
            2.1211E+02, 2.0096E+02, 1.9600E+02, 1.9308E+02, 1.9020E+02,
            1.8888E+02, 1.8756E+02, 1.8625E+02, 1.8495E+02, 1.8329E+02,
            1.7523E+02, 1.6752E+02, 1.6306E+02, 1.5418E+02, 1.4666E+02,
            1.3950E+02, 1.3270E+02, 1.2623E+02, 1.2055E+02, 1.1758E+02,
            1.1652E+02, 1.1548E+02, 1.1285E+02, 1.1029E+02, 1.0565E+02,
            1.0304E+02, 1.0211E+02, 1.0161E+02, 1.0110E+02, 1.0059E+02,
            9.7329E+01, 9.3326E+01, 8.8774E+01, 8.3939E+01, 7.9368E+01,
            7.6332E+01, 7.3559E+01, 7.1887E+01, 6.9068E+01, 6.6826E+01,
            6.6493E+01, 6.6161E+01, 6.5831E+01, 6.5503E+01, 6.5046E+01,
            6.4592E+01, 6.3631E+01, 6.2308E+01, 5.9925E+01, 5.7059E+01,
            5.4060E+01, 5.2990E+01, 5.1785E+01, 4.9259E+01, 4.7517E+01,
            4.6205E+01, 4.5290E+01, 4.4172E+01, 4.3125E+01, 4.2144E+01,
            4.1227E+01, 3.9730E+01, 3.8787E+01, 3.7792E+01, 3.7304E+01,
            3.6859E+01, 3.6419E+01, 3.6057E+01, 3.5698E+01, 3.4539E+01,
            3.3085E+01, 3.1693E+01, 2.7885E+01, 2.4658E+01, 2.252E+01,
            2.238E+01, 2.216E+01, 2.200E+01, 2.170E+01, 2.149E+01, 2.134E+01,
            2.123E+01, 2.114E+01, 2.106E+01, 2.098E+01, 2.077E+01, 2.068E+01,
            2.060E+01, 2.052E+01, 2.042E+01, 2.028E+01, 2.007E+01, 1.960E+01,
            1.939E+01, 1.920E+01, 1.908E+01, 1.796E+01, 1.776E+01, 1.756E+01,
            1.745E+01, 1.683E+01, 1.655E+01, 1.605E+01, 1.578E+01, 1.487E+01,
            1.473E+01, 1.460E+01, 1.447E+01, 1.425E+01, 1.405E+01, 1.355E+01,
            1.333E+01, 1.260E+01, 1.247E+01, 1.231E+01, 1.213E+01, 1.198E+01,
            1.182E+01, 1.171E+01, 1.159E+01, 1.127E+01, 1.105E+01, 1.080E+01,
            1.058E+01, 9.500E+00, 9.140E+00, 8.980E+00, 8.800E+00, 8.674E+00,
            8.524E+00, 8.300E+00, 8.130E+00, 7.970E+00, 7.840E+00, 7.740E+00,
            7.600E+00, 7.380E+00, 7.140E+00, 6.994E+00, 6.918E+00, 6.870E+00,
            6.835E+00, 6.811E+00, 6.792E+00, 6.776E+00, 6.760E+00, 6.742E+00,
            6.717E+00, 6.631E+00, 6.606E+00, 6.588E+00, 6.572E+00, 6.556E+00,
            6.539E+00, 6.515E+00, 6.482E+00, 6.432E+00, 6.360E+00, 6.280E+00,
            6.160E+00, 6.060E+00, 5.960E+00, 5.800E+00, 5.720E+00, 5.620E+00,
            5.530E+00, 5.488E+00, 5.410E+00, 5.380E+00, 5.320E+00, 5.210E+00,
            5.110E+00, 4.933E+00, 4.768E+00, 4.420E+00, 4.310E+00, 4.220E+00,
            4.000E+00, 3.882E+00, 3.712E+00, 3.543E+00, 3.142E+00, 2.884E+00,
            2.775E+00, 2.741E+00, 2.720E+00, 2.700E+00, 2.640E+00, 2.620E+00,
            2.590E+00, 2.550E+00, 2.470E+00, 2.330E+00, 2.273E+00, 2.217E+00,
            2.157E+00, 2.070E+00, 1.990E+00, 1.900E+00, 1.780E+00, 1.669E+00,
            1.588E+00, 1.520E+00, 1.444E+00, 1.410E+00, 1.381E+00, 1.331E+00,
            1.293E+00, 1.251E+00, 1.214E+00, 1.170E+00, 1.148E+00, 1.130E+00,
            1.116E+00, 1.104E+00, 1.092E+00, 1.078E+00, 1.035E+00, 1.021E+00,
            1.009E+00, 9.965E-01, 9.820E-01, 9.640E-01, 9.440E-01, 9.200E-01,
            8.800E-01, 8.200E-01, 7.200E-01, 6.250E-01, 5.950E-01, 5.550E-01,
            5.200E-01, 4.750E-01, 4.316E-01, 3.900E-01, 3.530E-01, 3.250E-01,
            3.050E-01, 2.800E-01, 2.550E-01, 2.312E-01, 2.096E-01, 1.900E-01,
            1.619E-01, 1.380E-01, 1.200E-01, 1.043E-01, 8.980E-02, 7.650E-02,
            6.520E-02, 5.550E-02, 4.730E-02, 4.030E-02, 3.440E-02, 2.930E-02,
            2.494E-02, 2.001E-02, 1.483E-02, 1.045E-02, 7.145E-03, 4.556E-03,
            2.500E-03, 1e-5
        ])
        self._fispec = np.zeros(361)
        self._first_res = 56
        self._last_res = 172

    def build_helios190(self):
        self._group_boundaries = np.array([
            2.0000E+07, 1.7000E+07, 1.4919E+07, 1.3380E+07,
            1.2000E+07, 1.0000E+07, 8.8250E+06, 7.7880E+06, 7.4082E+06,
            6.0653E+06, 5.2205E+06, 4.7237E+06, 4.4933E+06, 4.0657E+06,
            3.6788E+06, 3.1664E+06, 2.8650E+06, 2.7253E+06, 2.4660E+06,
            2.3650E+06, 2.3457E+06, 2.2313E+06,
            2.0189E+06, 1.8268E+06, 1.7377E+06, 1.5724E+06, 1.3534E+06,
            1.1649E+06, 1.0540E+06, 1.0026E+06, 8.2085E+05, 7.0651E+05,
            6.3928E+05, 6.0810E+05, 4.9787E+05, 4.2852E+05, 3.8774E+05,
            3.6883E+05, 3.0197E+05, 2.5991E+05,
            2.3518E+05, 2.2371E+05, 1.8316E+05, 1.4996E+05, 1.4264E+05,
            1.2907E+05, 1.1109E+05, 8.6517E+04, 6.7379E+04, 5.2474E+04,
            4.0868E+04, 3.6066E+04, 3.1828E+04, 2.8088E+04, 2.6058E+04,
            2.4788E+04, 2.1875E+04, 1.9305E+04,
            1.7036E+04, 1.5034E+04, 1.3268E+04, 1.1709E+04, 1.0333E+04,
            9.1188E+03, 8.0473E+03, 7.1017E+03, 6.2673E+03, 5.5308E+03,
            4.8810E+03, 4.3074E+03, 3.8013E+03, 3.3546E+03, 2.9604E+03,
            2.6126E+03, 2.3056E+03, 2.0347E+03,
            1.7956E+03, 1.5846E+03, 1.3984E+03, 1.2341E+03, 1.0891E+03,
            9.6112E+02, 8.4818E+02, 7.4852E+02, 6.6057E+02, 5.8295E+02,
            5.1445E+02, 4.5400E+02, 4.0065E+02, 3.5358E+02, 3.1203E+02,
            2.7536E+02, 2.4301E+02, 2.1445E+02,
            1.8926E+02, 1.6702E+02, 1.4739E+02, 1.3007E+02, 1.1479E+02,
            1.0130E+02, 8.9398E+01, 7.8893E+01, 6.9623E+01, 6.1442E+01,
            5.4222E+01, 4.7851E+01, 4.2229E+01, 3.7267E+01, 3.2888E+01,
            2.9023E+01, 2.5613E+01, 2.2603E+01,
            1.9947E+01, 1.7603E+01, 1.5536E+01, 1.3710E+01, 1.2099E+01,
            1.0677E+01, 9.4225E+00, 8.3153E+00, 7.3382E+00, 6.8680E+00,
            6.4760E+00, 5.7150E+00, 5.0435E+00, 4.4509E+00, 3.9279E+00,
            3.4663E+00, 3.0590E+00, 2.6996E+00,
            2.3824E+00, 2.1024E+00, 1.8554E+00, 1.7896E+00, 1.7261E+00,
            1.6592E+00, 1.5949E+00, 1.5246E+00, 1.4574E+00, 1.3806E+00,
            1.3079E+00, 1.2351E+00, 1.1664E+00, 1.1254E+00, 1.0987E+00,
            1.0722E+00, 1.0623E+00, 1.0525E+00,
            1.0427E+00, 1.0137E+00, 9.9200E-01, 9.7100E-01, 9.5065E-01,
            9.1000E-01, 8.7642E-01, 8.3368E-01, 7.8208E-01, 7.3000E-01,
            6.7000E-01, 6.2506E-01, 5.7000E-01, 5.3000E-01, 5.0323E-01,
            4.5000E-01, 4.1704E-01, 3.5767E-01,
            3.2063E-01, 3.0112E-01, 2.9074E-01, 2.7052E-01, 2.5103E-01,
            2.2769E-01, 1.8443E-01, 1.5230E-01, 1.4572E-01, 1.1157E-01,
            8.1968E-02, 6.7000E-02, 5.6922E-02, 5.0000E-02, 4.2755E-02,
            3.5500E-02, 3.0613E-02, 2.5500E-02,
            2.0492E-02, 1.2396E-02, 6.3247E-03, 2.2769E-03, 7.6022E-04,
            2.5399E-04, 1.0000E-04
        ])
        self._fispec = np.zeros(190)
        self._first_res = 63
        self._last_res = 125

    def build_helios45(self):
        self._group_boundaries = np.array([
            20000000.0000, 6065300.0000, 3678800.0000, 2231300.0000,
            1353400.0000, 820850.0000, 497870.0000, 183160.0000, 67379.0000,
            9118.8000, 2034.7000, 130.0700, 78.8930, 47.8510, 29.0230, 13.7100,
            12.0990, 8.3153, 7.3382, 6.4760, 5.7150, 5.0435, 4.4509, 3.9279,
            2.3824, 1.8554, 1.4574, 1.2351, 1.1254, 1.0722, 1.0137, 0.9710,
            0.9100, 0.7821, 0.6251, 0.3577, 0.2705, 0.1844, 0.1457, 0.1116,
            0.0820, 0.0569, 0.0428, 0.0306, 0.0124, 0.0001
        ])
        self._fispec = np.zeros(45)
        self._first_res = 9
        self._last_res = 22

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
            2.00000E-02, 1.50000E-02, 1.00000E-02, 5.00000E-03, 1.00000E-05
        ])
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
        self._batches = opts.ri_batches
        self._particles = opts.ri_particles
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
        from openmoc import SDSolver
        from openmoc.library_ce import LibraryCe
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
    lib_fname = 'jeff-3.2-wims69e-new.h5'
    # set_default_settings(batches=10, inactive=3, particles=50)
    opts_list = []

    # Options for generating U238
    opts_u238 = MicroMgXsOptions()
    opts_u238.nuclide = 'U238'
    opts_u238.has_res = True
    opts_u238.reference_dilution = 28.0
    opts_list.append(opts_u238)

    # Options for generating U235
    opts_u235 = MicroMgXsOptions()
    opts_u235.nuclide = 'U235'
    opts_u235.has_res = True
    opts_u235.has_resfis = True
    opts_u235.reference_dilution = 800.0
    opts_list.append(opts_u235)

    # Options for generating Pu239
    opts_pu239 = MicroMgXsOptions()
    opts_pu239.nuclide = 'Pu239'
    opts_pu239.has_res = True
    opts_pu239.has_resfis = True
    opts_pu239.reference_dilution = 700.0
    opts_list.append(opts_pu239)

    # Options for generating Pu240
    opts_pu240 = MicroMgXsOptions()
    opts_pu240.nuclide = 'Pu240'
    opts_pu240.has_res = True
    opts_pu240.reference_dilution = 2e3
    opts_list.append(opts_pu240)

    # Options for generating Pu241
    opts_pu241 = MicroMgXsOptions()
    opts_pu241.nuclide = 'Pu241'
    opts_pu241.has_res = True
    opts_pu241.has_resfis = True
    opts_pu241.reference_dilution = 1e4
    opts_list.append(opts_pu241)

    # Options for generating Pu242
    opts_pu242 = MicroMgXsOptions()
    opts_pu242.nuclide = 'Pu242'
    opts_pu242.has_res = True
    opts_pu242.reference_dilution = 1e5
    opts_list.append(opts_pu242)

    # Options for generating H1
    opts_h1 = MicroMgXsOptions()
    opts_h1.nuclide = 'H1'
    opts_list.append(opts_h1)

    # # Options for generating O16
    opts_o16 = MicroMgXsOptions()
    opts_o16.nuclide = 'O16'
    opts_list.append(opts_o16)

    lib = MicroMgXsLibrary(opts_list, lib_fname)
    lib.build_library_h5()
