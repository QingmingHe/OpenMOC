#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" brief description

Author: Qingming He
Email: he_qing_ming@foxmail.com
Date: 2016-10-27 14:46:45

Description
===========


"""
from openmoc import SDSolver
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
        self._dilutions = [5.0, 1e1, 15.0, 20.0, 25.0, 28.0, 30.0, 35.0, 40.0,
                           45.0, 50.0, 52.0, 60.0, 70.0, 80.0, 1e2, 2e2, 4e2,
                           6e2, 1e3, 1.2e3, 1e4, 1e10]
        self._n_dilution = len(self._dilutions)

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
            hflib = self._celib.get_nuclide(nuclide, self._temperature, emax,
                                            emin, self._has_res_fis)
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
                for idil in range(self._n_dilution):
                    xs_tot = sdnuc.getMgTotal(jg, idil)
                    self._res_sca[jg, inuc, idil] \
                        = sdnuc.getMgScatter(jg, idil)
                    self._res_abs[jg, inuc, idil] \
                        = xs_tot - self._res_sca[jg, inuc, idil]
                    if self._has_res_fis:
                        xs_typ = mglib.get_typical_xs(
                            nuclide, self._temperature, ig, 'nu')
                        self._res_nfi[jg, inuc, idil] \
                            = sdnuc.getMgFission(jg, idil) * xs_typ['nu']
                # Accumulate xs for pseudo nuclide
                self._res_abs[jg, self._n_nuclide, :] \
                    += self._res_abs[jg, inuc, :] * self._ratios[inuc]
                self._res_sca[jg, self._n_nuclide, :] \
                    += self._res_sca[jg, inuc, :] * self._ratios[inuc]
                if self._has_res_fis:
                    self._res_nfi[jg, self._n_nuclide, :] \
                        += self._res_nfi[jg, inuc, :] * self._ratios[inuc]

    def get_subp_one_nuc(self, ig, nuclide):
        mglib = self._mglib
        jg = ig - mglib.first_res
        inuc = self._nuclides.index(nuclide)

        # Fit subgroup parameters
        pt = ProbTable()
        pt.gc_factor = self._gc_factor[jg]
        pt.potential = self._potential[jg]
        pt.dilutions = self._dilutions
        pt.xs_abs = self._res_abs[jg, inuc, :]
        pt.xs_sca = self._res_sca[jg, inuc, :]
        if self._has_res_fis:
            pt.xs_nfi = self._res_nfi[jg, inuc, :]
        pt.fit()

        return {'lambda': pt.gc_factor, 'potential': pt.potential, 'sub_tot':
                pt.sub_tot, 'sub_abs': pt.sub_abs, 'sub_sca': pt.sub_sca,
                'sub_nfi': pt.sub_nfi, 'sub_wgt': pt.sub_wgt, 'n_band':
                pt.n_band}

if __name__ == '__main__':
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmc/openmc/micromgxs/',
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
