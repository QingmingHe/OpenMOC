#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from copy import deepcopy
from math import pi, sqrt
from openmoc.library_ce import LibraryCe


def _equal_vol_radius(r0, r1, n):
    v0 = pi * r0 ** 2
    v1 = pi * r1 ** 2
    v = (v1 - v0) / n
    r = []
    for i in range(1, n):
        r.append(sqrt((v0 + v * i) / pi))
    return r


class RMET21Nuclide(object):
    def __init__(self):
        self._name = None
        self._has_res = False
        self._has_res_fis = False
        self._num_dens = []

        # Use value given by user if possible
        self._awr = None
        self._potential = None
        self._temp = None

        # Shape = (n_case or n_region + 1, n_group)
        self._xs_abs = None
        self._xs_sca = None
        self._xs_fis = None
        self._xs_tot = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def has_res(self):
        return self._has_res

    @has_res.setter
    def has_res(self, has_res):
        self._has_res = has_res

    @property
    def has_res_fis(self):
        return self._has_res_fis

    @has_res_fis.setter
    def has_res_fis(self, has_res_fis):
        self._has_res_fis = has_res_fis

    @property
    def num_dens(self):
        return self._num_dens

    @num_dens.setter
    def num_dens(self, num_dens):
        self._num_dens = num_dens

    @property
    def awr(self):
        return self._awr

    @awr.setter
    def awr(self, awr):
        self._awr = awr

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = potential

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, temp):
        self._temp = temp

    @property
    def xs_abs(self):
        return self._xs_abs

    @xs_abs.setter
    def xs_abs(self, xs_abs):
        self._xs_abs = xs_abs

    @property
    def xs_sca(self):
        return self._xs_sca

    @xs_sca.setter
    def xs_sca(self, xs_sca):
        self._xs_sca = xs_sca

    @property
    def xs_fis(self):
        return self._xs_fis

    @xs_fis.setter
    def xs_fis(self, xs_fis):
        self._xs_fis = xs_fis

    @property
    def xs_tot(self):
        return self._xs_tot

    @xs_tot.setter
    def xs_tot(self, xs_tot):
        self._xs_tot = xs_tot


class RMET21Material(object):
    def __init__(self):
        self._nuclides = None
        self._num_dens = None
        self._temp = None

    def extend(self, material, ratio=1.0):
        if self._nuclides is None:
            self._nuclides = deepcopy(material.nuclides)
        else:
            self._nuclides.extend(material.nuclides)
        if self._num_dens is None:
            self._num_dens = [val * ratio for val in material.num_dens]
        else:
            self._num_dens.extend([val * ratio for val in material.num_dens])
        if self._temp is None:
            self._temp = material.temp

    @property
    def nuclides(self):
        return self._nuclides

    @nuclides.setter
    def nuclides(self, nuclides):
        self._nuclides = nuclides

    @property
    def num_dens(self):
        return self._num_dens

    @num_dens.setter
    def num_dens(self, num_dens):
        self._num_dens = num_dens

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, temp):
        self._temp = temp

    @property
    def n_nuclide(self):
        return len(self._nuclides)

    @property
    def n_res_nuc(self):
        n = 0
        for nuclide in self._nuclides:
            if nuclide.has_res:
                n += 1
        return n


class RMET21(object):
    def __init__(self):
        self._is_homo = True
        self._ni = 10000
        self._npft = 1000000
        self._grp_bnds = []
        self._emax = 1e4
        self._emin = 4.0

        # Only for heterogeneous problem
        self._materials = []
        self._radius = []
        self._nmesh = []

        self._input_fname = 'RMET21.DAT'
        self._output_fname = 'RMET21.OUT'
        self._cross_sections = os.getenv('OPENMC_CROSS_SECTIONS')
        self._rmet21 = 'rmet21'

        self._n_res_nuc = 0
        self._nuclides = []

    @property
    def is_homo(self):
        return self._is_homo

    @is_homo.setter
    def is_homo(self, is_homo):
        self._is_homo = is_homo

    @property
    def ni(self):
        return self._ni

    @ni.setter
    def ni(self, ni):
        self._ni = ni

    @property
    def npft(self):
        return self._npft

    @npft.setter
    def npft(self, npft):
        self._npft = npft

    @property
    def grp_bnds(self):
        return self._grp_bnds

    @grp_bnds.setter
    def grp_bnds(self, grp_bnds):
        self._grp_bnds = grp_bnds

    @property
    def emax(self):
        return self._emax

    @emax.setter
    def emax(self, emax):
        self._emax = emax

    @property
    def emin(self):
        return self._emin

    @emin.setter
    def emin(self, emin):
        self._emin = emin

    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, materials):
        self._materials = materials

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def nmesh(self):
        return self._nmesh

    @nmesh.setter
    def nmesh(self, nmesh):
        self._nmesh = nmesh

    def run(self):
        self._get_all_nuclides()
        self._prepare_lib()

        if self._is_homo:
            self._run_homo()
        else:
            self._run_heter()

    def read_output(self, fname):
        self._output_fname = fname
        self._get_all_nuclides()

        if self._is_homo:
            self._read_homo_output()
        else:
            self._read_heter_output()

    def _prepare_lib(self):
        inuc = 0
        libce = LibraryCe(self._cross_sections)
        for nuclide in self._nuclides:
            if nuclide.has_res:
                inuc += 1
                xs_fname = 'XS%i' % inuc
            else:
                xs_fname = '%s.XS' % nuclide.name

            attrs = libce.to_rmet21_xs_file(
                xs_fname, nuclide.name, nuclide.temp, self._npft,
                self._ni, self._emax, self._emin, nuclide.has_res_fis)
            if nuclide.potential is None:
                nuclide.potential = attrs['potential']
            if nuclide.awr is None:
                nuclide.awr = attrs['awr']

    def _run_homo(self):
        self._write_homo_input()
        os.system(self._rmet21)
        self._read_homo_output()

    def _run_heter(self):
        self._write_heter_input()
        os.system(self._rmet21)
        self._read_heter_output()

    def _write_homo_input(self):
        temp = self._materials[0].temp

        with open(self._input_fname, 'w') as f:
            f.write('1\n')
            f.write('comment\n')
            f.write('0\n')
            f.write('0\n')
            f.write('%i %i %i %i\n' % (len(self._grp_bnds), len(self._nuclides),
                                       self._n_res_nuc, self._ni))
            f.write('0.0\n')
            f.write('%i\n' % len(self._nuclides))
            f.write(' '.join([str(val) for val in
                              range(1, len(self._nuclides) + 1)]))
            f.write('\n')
            f.write(' '.join(['%e' % nuclide.num_dens[0] for nuclide in
                              self._nuclides]))
            f.write('\n')
            f.write('%f\n' % temp)
            for nuclide in self._nuclides:
                f.write('%s\n' % nuclide.name)
                f.write('%f\n' % nuclide.awr)
                if nuclide.has_res:
                    f.write('1\n')
                else:
                    f.write('0\n')
                f.write('%f\n' % nuclide.potential)
                if not nuclide.has_res:
                    f.write('0.0 0.0\n')
            f.write(' '.join([str(val) for val in self._grp_bnds]))
            f.write('\n')
            f.write('0\n')
            f.write('%i\n' % len(self._nuclides[0].num_dens))
            for icase in range(1, len(self._nuclides[0].num_dens)):
                f.write(' '.join(['%e' % nuclide.num_dens[icase] for nuclide in
                                  self._nuclides]))
                f.write('\n')
                f.write('%f\n' % temp)

    def _get_all_nuclides(self):
        # Nuclides with same name should not share memory
        materials = []
        for material in self._materials:
            if material in materials:
                raise Exception('Same material occur')
            materials.append(material)
            nuclides = []
            for inuc, nuclide in enumerate(material.nuclides):
                nuclides.append(deepcopy(nuclide))
                if nuclides[inuc].temp is None:
                    nuclides[inuc].temp = material.temp
            material.nuclides = nuclides

        # Get nuclides with different names and different temperatures
        self._nuclides = []
        self._n_res_nuc = 0
        for material in self._materials:
            for nuclide in material.nuclides:
                find = False
                for nuc in self._nuclides:
                    if nuc.name == nuclide.name and nuc.temp == nuclide.temp:
                        find = True
                if not find:
                    self._nuclides.append(deepcopy(nuclide))
                    if nuclide.has_res:
                        self._n_res_nuc += 1

    def _write_heter_input(self):
        nmat = len(self._materials)
        nreg = sum(self._nmesh)
        nz = nmat
        ismax = 0
        nirmax = 0
        for material in self._materials:
            if ismax < material.n_nuclide:
                ismax = material.n_nuclide
            if nirmax < material.n_res_nuc:
                nirmax = material.n_res_nuc

        with open(self._input_fname, 'w') as f:
            f.write('1\n')
            f.write('comment\n')
            f.write('1\n')
            f.write('%i %i %i %i\n' % (len(self._grp_bnds), len(self._nuclides),
                                       self._n_res_nuc, self._ni))
            f.write('2.438\n')
            f.write('%i %i %i %i %i\n' % (nmat, nreg, nz, ismax, nirmax))

            # Material data
            for material in self._materials:
                nuclides = {}
                for jnuc, nuclide in enumerate(material.nuclides):
                    for inuc, nuc in enumerate(self._nuclides):
                        if nuclide.name == nuc.name and \
                           nuclide.temp == nuc.temp:
                            nuclides[inuc+1] = material.num_dens[jnuc]
                            break
                f.write('%i\n' % len(material.nuclides))
                for inuc in nuclides:
                    f.write('%i ' % inuc)
                f.write('\n')
                for inuc in nuclides:
                    f.write('%e ' % nuclides[inuc])
                f.write('\n')
                f.write('%f' % material.temp)
                f.write('\n')

            # Nuclide data
            for nuclide in self._nuclides:
                f.write('%s\n' % nuclide.name)
                f.write('%f\n' % nuclide.awr)
                if nuclide.has_res:
                    f.write('1\n')
                else:
                    f.write('0\n')
                f.write('%f\n' % nuclide.potential)
                if not nuclide.has_res:
                    f.write('0.0 0.0\n')

            # Geometry data
            for imat in range(len(self._materials)):
                f.write(' '.join([str(imat+1) for ireg in
                                  range(self._nmesh[imat])]))
                f.write(' ')
            f.write('\n')
            r = _equal_vol_radius(0.0, self._radius[0], self._nmesh[0])
            f.write(' '.join([str(val) for val in r]))
            f.write(' ')
            for imat in range(len(self._materials) - 1):
                f.write('%f ' % self._radius[imat])
                r = _equal_vol_radius(self._radius[imat],
                                      self._radius[imat+1],
                                      self._nmesh[imat+1])
                f.write(' '.join([str(val) for val in r]))
                f.write(' ')
            f.write('%f\n' % self._radius[-1])
            f.write(' '.join([str(val) for val in self._nmesh]))
            f.write('\n')

            f.write(' '.join([str(val) for val in self._grp_bnds]))
            f.write('\n0\n1\n')

    def _read_homo_output(self):
        # Initialize XS
        ncase = len(self._nuclides[0].num_dens)
        ng = len(self._grp_bnds) - 1
        nnuc = len(self._nuclides)
        xs_abs = np.zeros((nnuc, ncase, ng))
        xs_sca = np.zeros((nnuc, ncase, ng))
        xs_fis = np.zeros((nnuc, ncase, ng))

        with open(self._output_fname) as f:
            for aline in f:
                if 'TEMPERATURE' in aline:
                    break
            for icase in range(ncase):
                for aline in f:
                    if 'TEMPERATURE' in aline:
                        break
                if icase == 0:
                    n = 7 + nnuc
                else:
                    n = 5 + nnuc
                for i in range(n):
                    f.next()

                # Read XS
                for ig in range(ng):
                    f.next()
                    f.next()
                    for inuc in range(nnuc):
                        num_dens = self._nuclides[inuc].num_dens[icase]
                        aline = f.next()
                        xs_abs[inuc, icase, ig] = float(aline[15:27]) / num_dens
                        xs_fis[inuc, icase, ig] = float(aline[27:39]) / num_dens
                        xs_sca[inuc, icase, ig] = float(aline[39:51]) / num_dens
                    f.next()
                    f.next()

        for inuc, nuclide in enumerate(self._materials[0].nuclides):
            nuclide.xs_abs = xs_abs[inuc, :, :]
            nuclide.xs_fis = xs_fis[inuc, :, :]
            nuclide.xs_sca = xs_sca[inuc, :, :]
            nuclide.xs_tot = nuclide.xs_abs + nuclide.xs_sca

    def _read_heter_output(self):
        ng = len(self._grp_bnds) - 1
        nz = len(self._materials)
        nnuc = len(self._nuclides)

        with open(self._output_fname) as f:
            for aline in f:
                if 'GRO ' in aline:
                    break
            for aline in f:
                if 'CELL' in aline:
                    break
            for i in range(nnuc-1):
                f.next()

            for ig in range(ng):
                for iz in range(nz):
                    material = self._materials[iz]
                    for ireg in range(self._nmesh[iz]+1):
                        f.next()
                        for inuc, nuclide in enumerate(material.nuclides):
                            if nuclide.xs_abs is None:
                                nuclide.xs_abs = np.zeros((self._nmesh[iz]+1,
                                                           ng))
                                nuclide.xs_sca = np.zeros((self._nmesh[iz]+1,
                                                           ng))
                                nuclide.xs_tot = np.zeros((self._nmesh[iz]+1,
                                                           ng))
                                nuclide.xs_fis = np.zeros((self._nmesh[iz]+1,
                                                           ng))
                            aline = f.next()
                            f.next()
                            num_dens = material.num_dens[inuc]
                            nuclide.xs_abs[ireg, ig] \
                                = float(aline[18:30]) / num_dens
                            nuclide.xs_fis[ireg, ig] \
                                = float(aline[30:42]) / num_dens
                            nuclide.xs_sca[ireg, ig] \
                                = float(aline[42:54]) / num_dens
                            nuclide.xs_tot[ireg, ig] \
                                = nuclide.xs_sca[ireg, ig] + \
                                nuclide.xs_abs[ireg, ig]
                        f.next()
                for inuc in range(len(self._nuclides) + 1):
                    f.next()

    def _build_xs_table(self, imat, nuc_indexes):
        dilutions = [5.0, 1e1, 15.0, 20.0, 25.0, 28.0, 30.0, 35.0, 40.0,
                     45.0, 50.0, 52.0, 60.0, 70.0, 80.0, 1e2, 2e2, 4e2,
                     6e2, 1e3, 1.2e3, 1e4, 1e10]

        material = RMET21Material()
        material.temp = self._materials[imat].temp
        for inuc, nuc in enumerate(self._materials[imat].nuclides):
            if inuc in nuc_indexes:
                nuclide = RMET21Nuclide()
                nuclide.name = self._materials[imat].nuclides[inuc].name
                nuclide.has_res = True
                nuclide.has_res_fis = nuc.has_res_fis
                nuclide.num_dens = [self._materials[imat].num_dens[inuc] for i
                                    in range(len(dilutions))]
                material.nuclides.append(nuclide)

        h1 = RMET21Nuclide()
        h1.name = 'H1'
        h1.num_dens = dilutions
        h1.potential = 1.0
        material.nuclides.append(h1)

        solver = RMET21()
        solver.materials = [material]
        solver.grp_bnds = self.grp_bnds
        solver.run()

        return solver

    def correct_homo(self, imat, nuc_indexes, materials, ratios):
        if self._is_homo:
            raise Exception('self should be heterogeneous')

        # Build resonance XS table
        hsolver0 = self._build_xs_table(imat, nuc_indexes)

        # Get dilution XS

        # Treat as resonant nuclides
        hsolver1 = RMET21()
        hsolver1.materials = [deepcopy(self._materials[imat])]
        for jmat, material in enumerate(materials):
            material1 = deepcopy(material)
            hsolver1.materials[0].extend(material1, ratios[jmat])
        hsolver1.grp_bnds = self.grp_bnds
        hsolver1.run()


def test_homo():
    u238 = RMET21Nuclide()
    u238.name = 'U238'
    u238.has_res = True
    u238.num_dens = [1.0, 1.0]

    h1 = RMET21Nuclide()
    h1.name = 'H1'
    h1.num_dens = [10.0, 28.0]
    h1.potential = 1.0

    mat = RMET21Material()
    mat.nuclides = [u238, h1]
    mat.temp = 293.6

    rmet21 = RMET21()
    rmet21.materials = [mat]
    rmet21.grp_bnds = [9118.0, 5530.0, 3519.1, 2239.45, 1425.1, 906.989,
                       367.262, 148.728, 75.5014, 48.052, 27.70, 15.968, 9.877,
                       4.0]

    rmet21.run()

    u238 = mat.nuclides[0]
    print u238.xs_abs[1, :]


def test_heter():
    u235 = RMET21Nuclide()
    u235.name = 'U235'
    u235.has_res = True
    u235.has_res_fis = True

    u238 = RMET21Nuclide()
    u238.name = 'U238'
    u238.has_res = True

    o16 = RMET21Nuclide()
    o16.name = 'O16'
    o16.has_res = True

    h1 = RMET21Nuclide()
    h1.name = 'H1'
    h1.has_res = True

    fuel = RMET21Material()
    fuel.nuclides = [u235, u238, o16]
    fuel.num_dens = [7.18132e-4, 2.21546e-2, 4.57642e-2]
    fuel.temp = 600.0

    water = RMET21Material()
    water.nuclides = [o16, h1]
    water.num_dens = [2.48112E-02, 4.96224E-02]
    water.temp = 600.0

    rmet21 = RMET21()
    rmet21.is_homo = False
    rmet21.materials = [fuel, water]
    rmet21.radius = [0.4095, 0.720322]
    rmet21.nmesh = [20, 1]
    rmet21.grp_bnds = [9118.0, 5530.0, 3519.1, 2239.45, 1425.1, 906.989,
                       367.262, 148.728, 75.5014, 48.052, 27.70, 15.968, 9.877,
                       4.0]

    rmet21.run()

if __name__ == '__main__':
    test_homo()
    # test_heter()
