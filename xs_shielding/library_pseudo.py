#!/usr/bin/env python
# -*- coding: utf-8 -*-
from openmoc import SDSolver, SDNuclide
from openmoc.library_ce import LibraryCe
import numpy as np
from prob_table import ProbTable
import os
from library_micro import LibraryMicro


class LibraryPseudo(object):

    def __init__(self):
        self._nuclides = None
        self._densities = None
        self._temperature = None
        self._cross_sections = None
        self._mglib = None
        self._dilutions = [1e1, 15.0, 20.0, 25.0, 28.0, 30.0, 35.0, 40.0,
                           45.0, 50.0, 52.0, 60.0, 70.0, 80.0, 90.0, 1e2, 2e2,
                           4e2, 6e2, 8e2, 1e3, 1.2e3, 1e4, 1e10]
        self._n_dilution = len(self._dilutions)
        self._temps = None
        self._find_nearest_temp = False
        self._max_n_band = 6
        self._subx_method = 0

        self._n_nuclide = 0
        self._has_res_fis = False
        self._celib = None
        self._ratios = None
        self._res_abs = None
        self._res_sca = None
        self._res_nfi = None
        self._gc_factor = None
        self._potential = None

    @property
    def nuclides(self):
        return self._nuclides

    @nuclides.setter
    def nuclides(self, nuclides):
        self._nuclides = nuclides
        self._n_nuclide = len(nuclides)

    @property
    def find_nearest_temps(self):
        return self._find_nearest_temps

    @find_nearest_temps.setter
    def find_nearest_temps(self, find_nearest_temps):
        self._find_nearest_temps = find_nearest_temps
        self._n_find_nearest_temp = len(find_nearest_temps)

    @property
    def densities(self):
        return self._densities

    @densities.setter
    def densities(self, densities):
        self._densities = densities
        self._ratios = np.array(densities) / np.sum(densities)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature

    @property
    def subx_method(self):
        return self._subx_method

    @subx_method.setter
    def subx_method(self, subx_method):
        self._subx_method = subx_method

    @property
    def max_n_band(self):
        return self._max_n_band

    @max_n_band.setter
    def max_n_band(self, max_n_band):
        self._max_n_band = max_n_band

    @property
    def cross_sections(self):
        return self._cross_sections

    @cross_sections.setter
    def cross_sections(self, cross_sections):
        self._cross_sections = cross_sections
        self._celib = LibraryCe(cross_sections)

    @property
    def has_res_fis(self):
        return self._has_res_fis

    @has_res_fis.setter
    def has_res_fis(self, has_res_fis):
        self._has_res_fis = has_res_fis

    @property
    def mglib(self):
        return self._mglib

    @mglib.setter
    def mglib(self, mglib):
        self._mglib = mglib

    @property
    def temps(self):
        return self._temps

    @temps.setter
    def temps(self, temps):
        self._temps = temps

    @property
    def dilutions(self):
        return self._dilutions

    @dilutions.setter
    def dilutions(self, dilutions):
        self._dilutions = dilutions
        self._n_dilution = len(dilutions)

    def _init_params(self):
        # Whether has resonance fission
        self._has_res_fis = False
        for nuclide in self._nuclides:
            if self._mglib.has_resfis(nuclide):
                self._has_res_fis = True
                break

        # Goldstein-Cohen factor and potential cross sections of pseudo nuclide
        n_res = self._mglib.last_res - self._mglib.first_res
        self._gc_factor = np.zeros(n_res)
        self._potential = np.zeros(n_res)
        for ig in range(self._mglib.first_res, self._mglib.last_res):
            jg = ig - self._mglib.first_res
            for inuc, nuclide in enumerate(self._nuclides):
                xs = self._mglib.get_typical_xs(nuclide, self._temperature, ig,
                                                'lambda', 'potential')
                self._gc_factor[jg] += self._ratios[inuc] * xs['lambda']
                self._potential[jg] += self._ratios[inuc] * xs['potential']

    def make(self):
        if self._temps is None:
            self.make_ave_temp()
        else:
            self.make_multi_temps()

    def make_multi_temps(self):
        self._init_params()
        mglib = self._mglib
        n_temp = len(self._temps)

        # Get energy boundaries for slowing down calculation
        emax = mglib.group_boundaries[self._mglib.first_res]
        emin = mglib.group_boundaries[self._mglib.last_res]

        # Initialize slowing down solver
        sd = SDSolver()
        sd.setErgGrpBnd(
            mglib.group_boundaries[mglib.first_res:mglib.last_res+1])
        sd.setNumNuclide(self._n_nuclide + 1)
        sd.setSolErgBnd(emin, emax)

        # Initialize resonant nuclides of slowing down solver
        for inuc, nuclide in enumerate(self._nuclides):
            hflib = self._celib.get_nuclide(
                nuclide, self._temperature, emax, emin, self._has_res_fis,
                find_nearest_temp=self._find_nearest_temp)
            sdnuc = sd.getNuclide(inuc)
            sdnuc.setName(nuclide)
            sdnuc.setHFLibrary(hflib)
            sdnuc.setNumDens(np.zeros(self._n_dilution) + self._ratios[inuc])

        # Initialize background nuclide of slowing down solver
        sdnuc = sd.getNuclide(self._n_nuclide)
        sdnuc.setName('H1')
        sdnuc.setAwr(0.9991673)
        sdnuc.setPotential(1.0)
        sdnuc.setNumDens(np.array(self._dilutions))

        # Solve slowing down equation
        sd.computeFlux()
        sd.computeMgXs()

        # Initialize resonance xs table
        n_res = mglib.last_res - mglib.first_res
        self._res_abs = np.zeros((n_res, self._n_nuclide+1, n_temp+1,
                                  self._n_dilution))
        self._res_sca = np.zeros((n_res, self._n_nuclide+1, n_temp+1,
                                  self._n_dilution))
        if self._has_res_fis:
            self._res_nfi = np.zeros((n_res, self._n_nuclide+1, n_temp+1,
                                      self._n_dilution))

        # Obtain resonance xs table at all temperatures
        for inuc, nuclide in enumerate(self._nuclides):
            for itemp in range(n_temp+1):
                if itemp == n_temp:
                    temp = self._temperature
                    sdnuc = sd.getNuclide(inuc)
                else:
                    temp = self._temps[itemp]
                    hflib = self._celib.get_nuclide(
                        nuclide, temp, emax, emin, self._has_res_fis,
                        find_nearest_temp=self._find_nearest_temp)
                    sdnuc = SDNuclide(nuclide)
                    sdnuc.setHFLibrary(hflib)
                    sdnuc.setNumDens(np.zeros(self._n_dilution) +
                                     self._ratios[inuc])
                    sd.computeMgXs(sdnuc)
                for ig in range(mglib.first_res, mglib.last_res):
                    jg = ig - mglib.first_res
                    for idlt in range(self._n_dilution):
                        xs_tot = sdnuc.getMgTotal(jg, idlt)
                        self._res_sca[jg, inuc, itemp, idlt] \
                            = sdnuc.getMgScatter(jg, idlt)
                        self._res_abs[jg, inuc, itemp, idlt] \
                            = xs_tot - self._res_sca[jg, inuc, itemp, idlt]
                        if self._has_res_fis:
                            xs_typ = mglib.get_typical_xs(
                                nuclide, temp, ig, 'nu')
                            self._res_nfi[jg, inuc, itemp, idlt] \
                                = sdnuc.getMgFission(jg, idlt) * xs_typ['nu']

            # Accumulate xs for pseudo nuclide
            self._res_abs[:, self._n_nuclide, :, :] \
                += self._res_abs[:, inuc, :, :] * self._ratios[inuc]
            self._res_sca[:, self._n_nuclide, :, :] \
                += self._res_sca[:, inuc, :, :] * self._ratios[inuc]
            if self._has_res_fis:
                self._res_nfi[:, self._n_nuclide, :, :] \
                    += self._res_nfi[:, inuc, :, :] * self._ratios[inuc]

        del sd

    def make_ave_temp(self):
        self._init_params()
        mglib = self._mglib

        # Get energy boundaries for slowing down calculation
        emax = mglib.group_boundaries[self._mglib.first_res]
        emin = mglib.group_boundaries[self._mglib.last_res]

        # Initialize slowing down solver
        sd = SDSolver()
        sd.setErgGrpBnd(
            mglib.group_boundaries[mglib.first_res:mglib.last_res+1])
        sd.setNumNuclide(self._n_nuclide + 1)
        sd.setSolErgBnd(emin, emax)

        # Initialize resonant nuclides of slowing down solver
        for inuc, nuclide in enumerate(self._nuclides):
            hflib = self._celib.get_nuclide(
                nuclide, self._temperature, emax, emin, self._has_res_fis,
                find_nearest_temp=self._find_nearest_temp)
            sdnuc = sd.getNuclide(inuc)
            sdnuc.setName(nuclide)
            sdnuc.setHFLibrary(hflib)
            sdnuc.setNumDens(np.zeros(self._n_dilution) + self._ratios[inuc])

        # Initialize background nuclide of slowing down solver
        sdnuc = sd.getNuclide(self._n_nuclide)
        sdnuc.setName('H1')
        sdnuc.setAwr(0.9991673)
        sdnuc.setPotential(1.0)
        sdnuc.setNumDens(np.array(self._dilutions))

        # Solve slowing down equation
        sd.computeFlux()
        sd.computeMgXs()

        # Obtain cross sections table
        n_res = mglib.last_res - mglib.first_res
        self._res_abs = np.zeros((n_res, self._n_nuclide+1, self._n_dilution))
        self._res_sca = np.zeros((n_res, self._n_nuclide+1, self._n_dilution))
        if self._has_res_fis:
            self._res_nfi = np.zeros((n_res, self._n_nuclide+1,
                                      self._n_dilution))
        for ig in range(mglib.first_res, mglib.last_res):
            jg = ig - mglib.first_res
            for inuc, nuclide in enumerate(self._nuclides):
                sdnuc = sd.getNuclide(inuc)
                for idlt in range(self._n_dilution):
                    xs_tot = sdnuc.getMgTotal(jg, idlt)
                    self._res_sca[jg, inuc, idlt] \
                        = sdnuc.getMgScatter(jg, idlt)
                    self._res_abs[jg, inuc, idlt] \
                        = xs_tot - self._res_sca[jg, inuc, idlt]
                    if self._has_res_fis:
                        xs_typ = mglib.get_typical_xs(
                            nuclide, self._temperature, ig, 'nu')
                        self._res_nfi[jg, inuc, idlt] \
                            = sdnuc.getMgFission(jg, idlt) * xs_typ['nu']
                # Accumulate xs for pseudo nuclide
                self._res_abs[jg, self._n_nuclide, :] \
                    += self._res_abs[jg, inuc, :] * self._ratios[inuc]
                self._res_sca[jg, self._n_nuclide, :] \
                    += self._res_sca[jg, inuc, :] * self._ratios[inuc]
                if self._has_res_fis:
                    self._res_nfi[jg, self._n_nuclide, :] \
                        += self._res_nfi[jg, inuc, :] * self._ratios[inuc]

        del sd

    def get_res_tbl(self, ig, nuclide, itemp=None):
        jg = ig - self._mglib.first_res
        inuc = self._nuclides.index(nuclide)

        if self._temps is None:
            res_abs = self._res_abs[jg, inuc, :]
            res_sca = self._res_sca[jg, inuc, :]
            res_tot = res_abs + res_sca
            if self._has_res_fis:
                res_nfi = self._res_nfi[jg, inuc, :]
            else:
                res_nfi = None
        else:
            if itemp is None:
                raise Exception('itemp should be given')
            res_abs = self._res_abs[jg, inuc, itemp, :]
            res_sca = self._res_sca[jg, inuc, itemp, :]
            res_tot = res_abs + res_sca
            if self._has_res_fis:
                res_nfi = self._res_nfi[jg, inuc, itemp, :]
            else:
                res_nfi = None

        return {'res_tot': res_tot, 'res_abs': res_abs, 'res_sca': res_sca,
                'res_nfi': res_nfi, 'dilutions': self._dilutions, 'lambda':
                self._gc_factor[jg], 'potential': self._potential[jg]}

    def get_subp_one_nuc(self, ig, nuclide):
        mglib = self._mglib
        jg = ig - mglib.first_res
        inuc = self._nuclides.index(nuclide)

        if self._temps is None:
            # Fit subgroup parameters
            pt = ProbTable()
            pt.subx_method = self._subx_method
            pt.gc_factor = self._gc_factor[jg]
            pt.potential = self._potential[jg]
            pt.dilutions = self._dilutions
            pt.xs_abs = self._res_abs[jg, inuc, :]
            pt.xs_sca = self._res_sca[jg, inuc, :]
            if self._has_res_fis:
                pt.xs_nfi = self._res_nfi[jg, inuc, :]
            pt.fit(n_band_end=self._max_n_band+1)

            return {'lambda': pt.gc_factor, 'potential': pt.potential,
                    'sub_tot': pt.sub_tot, 'sub_abs': pt.sub_abs, 'sub_sca':
                    pt.sub_sca, 'sub_nfi': pt.sub_nfi, 'sub_wgt': pt.sub_wgt,
                    'sub_int': pt.sub_int,
                    'n_band': pt.n_band}

        else:
            n_temp = len(self._temps)
            pt = ProbTable()
            pt.gc_factor = self._gc_factor[jg]
            pt.potential = self._potential[jg]
            pt.dilutions = self._dilutions
            pt.xs_abs = self._res_abs[jg, inuc, n_temp, :]
            pt.xs_sca = self._res_sca[jg, inuc, n_temp, :]
            if self._has_res_fis:
                pt.xs_nfi = self._res_nfi[jg, inuc, n_temp, :]
            if self._has_res_fis:
                n_xxx = 4
            else:
                n_xxx = 3
            xs_xxx = np.zeros((n_temp * n_xxx, self._n_dilution))
            i = 0
            check_xxx_idx = []
            for itemp in range(n_temp):
                xs_xxx[i, :] = self._res_abs[jg, inuc, itemp, :]
                xs_xxx[i+1, :] = self._res_sca[jg, inuc, itemp, :]
                xs_xxx[i+2, :] = xs_xxx[i, :] + xs_xxx[i+1, :]
                if self._has_res_fis:
                    xs_xxx[i+3, :] = self._res_nfi[jg, inuc, itemp, :]
                check_xxx_idx.extend([i+1, i+2])
                i += n_xxx
            pt.xs_xxx = xs_xxx
            pt.check_xxx_idx = check_xxx_idx
            pt.fit(n_band_end=self._max_n_band+1)

            pt_ave = {}
            pt_ave['lambda'] = pt.gc_factor
            pt_ave['potential'] = pt.potential
            pt_ave['sub_tot'] = pt.sub_tot
            pt_ave['sub_abs'] = pt.sub_abs
            pt_ave['sub_sca'] = pt.sub_sca
            pt_ave['sub_nfi'] = pt.sub_nfi
            pt_ave['sub_wgt'] = pt.sub_wgt
            pt_ave['sub_int'] = pt.sub_int
            pt_ave['n_band'] = pt.n_band
            pts = []
            i = 0
            for itemp in range(n_temp):
                pts.append({})
                pts[itemp]['lambda'] = pt.gc_factor
                pts[itemp]['potential'] = pt.potential
                pts[itemp]['n_band'] = pt.n_band
                if pt.n_band > 1:
                    pts[itemp]['sub_wgt'] = pt.sub_wgt
                    pts[itemp]['sub_abs'] = pt.sub_xxx[i, :]
                    pts[itemp]['sub_sca'] = pt.sub_xxx[i+1, :]
                    pts[itemp]['sub_tot'] = pt.sub_xxx[i+2, :]
                    if self._has_res_fis:
                        pts[itemp]['sub_nfi'] = pt.sub_xxx[i+3, :]
                i += n_xxx

            return pt_ave, pts

if __name__ == '__main__':
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmoc/micromgxs/',
                         'jeff-3.2-wims69e.h5')
    mglib = LibraryMicro()
    mglib.load_from_h5(fname)

    plib = LibraryPseudo()
    plib.nuclides = ['U238']
    plib.densities = [1.0]
    plib.temperature = 293.6
    plib.cross_sections = os.getenv('OPENMC_CROSS_SECTIONS')
    plib.mglib = mglib

    plib.make()

    for ig in range(mglib.first_res, mglib.last_res):
        subp = plib.get_subp_one_nuc(ig, 'U238')
        print ig, subp['n_band']
