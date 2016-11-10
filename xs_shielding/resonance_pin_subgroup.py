#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import openmoc
from openmoc.process import get_scalar_fluxes
from math import ceil
import h5py
from library_pseudo import LibraryPseudo
from prob_table import adjust_sub_level, unify_sub_wgt, comp_sub_level
from copy import deepcopy

PINCELLBOX = 1
PINCELLCYL = 2
openmoc.log.set_log_level('TITLE')
_openmoc_opts = openmoc.options.Options()


class Material(object):
    def __init__(self, temperature=None, nuclides=None, densities=None,
                 name=None):
        self.temperature = temperature
        self.nuclides = nuclides  # Nuclide names
        self.densities = densities  # Nuclide densities
        self.name = name  # Material name


class PinCell(object):
    def __init__(self, pin_type=None, radii=None, pitch=None, materials=None,
                 mat_fill=None, ave_temp=None):
        # Geometry definition
        self._pin_type = pin_type
        self._radii = radii
        self._pitch = pitch

        # Material definition (Matieral list)
        self._materials = materials

        # Materials in material regions
        self._mat_fill = mat_fill

        # Average temperature of fuel region
        self._ave_temp = ave_temp

    @property
    def pin_type(self):
        return self._pin_type

    @pin_type.setter
    def pin_type(self, pin_type):
        self._pin_type = pin_type

    @property
    def radii(self):
        return self._radii

    @radii.setter
    def radii(self, radii):
        self._radii = radii

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = pitch

    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, materials):
        self._materials = materials

    @property
    def mat_fill(self):
        return self._mat_fill

    @mat_fill.setter
    def mat_fill(self, mat_fill):
        self._mat_fill = mat_fill

    @property
    def ave_temp(self):
        return self._ave_temp

    @ave_temp.setter
    def ave_temp(self, ave_temp):
        self._ave_temp = ave_temp


class ResonancePinSubgroup(object):

    def __init__(self, pin_cell=None, pin_solver=None, micro_lib=None,
                 use_pseudo_lib=False, cross_sections=None, first_calc_g=None,
                 last_calc_g=None, pin_cell_ave=None):
        # Given by user
        self._pin_cell = pin_cell
        self._pin_solver = pin_solver
        self._micro_lib = micro_lib
        self._use_pseudo_lib = use_pseudo_lib
        self._cross_sections = cross_sections
        self._first_calc_g = first_calc_g
        self._last_calc_g = last_calc_g
        self._pin_cell_ave = pin_cell_ave

        self._pseudo_lib = None
        self._pseudo_libs = None
        self._resnuc_xs = None
        self._n_res_reg = None

    def solve_partial_xs_fit_var(self):
        if self._pin_cell.ave_temp is None:
            raise Exception('ave_temp of PinCell should be given')

        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Initialize pseudo library at all temperatures
        res_nuc = None
        res_nuc_temps = []
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_nuc_temps.append(mat.temperature)
                    break
        self._init_plib_one_nuc_multi_temp(res_nuc, 1.0, res_nuc_temps,
                                           self._pin_cell.ave_temp)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get subgroup parameters at all temperature
            pt_ave, pts = self._pseudo_lib.get_subp_one_nuc(ig, res_nuc)
            n_band = pt_ave['n_band']

            print(ig, n_band)
            if n_band > 1:
                sub_flux = np.zeros((n_band, n_region))
                for ib in range(n_band):
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * pt_ave['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * pt_ave['sub_sca'][ib] * \
                                    (1.0 - pt_ave['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    pt_ave['lambda'] * pt_ave['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * pt_ave['sub_wgt'][ib]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pts=pts)

    def solve_partial_xs_fit(self):
        if self._pin_cell.ave_temp is None:
            raise Exception('ave_temp of PinCell should be given')

        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Initialize pseudo library at all temperatures
        res_nuc = None
        res_nuc_temps = []
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_nuc_temps.append(mat.temperature)
                    break
        self._init_plib_one_nuc_multi_temp(res_nuc, 1.0, res_nuc_temps,
                                           self._pin_cell.ave_temp)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get subgroup parameters at all temperature
            pt_ave, pts = self._pseudo_lib.get_subp_one_nuc(ig, res_nuc)
            n_band = pt_ave['n_band']

            print(ig, n_band)
            if n_band > 1:
                sub_flux = np.zeros((n_band, n_region))
                for ib in range(n_band):
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * pts[ireg]['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * pts[ireg]['sub_sca'][ib] * \
                                    (1.0 - pts[ireg]['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    pts[ireg]['lambda'] * pts[ireg]['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * pt_ave['sub_wgt'][ib]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pts=pts)

    def solve_correlation_variant(self):
        if self._pin_cell.ave_temp is None:
            raise Exception('ave_temp of PinCell should be given')

        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Initialize pseudo library at all temperatures
        res_nuc = None
        res_nuc_temps = []
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_nuc_temps.append(mat.temperature)
                    break
        self._init_plib_one_nuc(res_nuc, 1.0, self._pin_cell.ave_temp)
        self._init_plib_temp_as_nuc(res_nuc, 1.0, res_nuc_temps,
                                    self._pin_cell.ave_temp)

        # Whether has resonance fission
        has_resfis = self._micro_lib.has_resfis(res_nuc)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get subgroup parameters at all temperature
            pt_ave = self._pseudo_lib.get_subp_one_nuc(ig, res_nuc)
            pts = []
            for itemp, temp in enumerate(res_nuc_temps):
                plib = self._pseudo_libs[itemp]
                pts.append(plib.get_subp_one_nuc(ig, res_nuc))
            n_band = pt_ave['n_band']

            print(ig, n_band)
            if n_band > 1:
                sub_flux = np.zeros((n_band, n_region))
                for ib in range(n_band):
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * pt_ave['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * pt_ave['sub_sca'][ib] * \
                                    (1.0 - pt_ave['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    pt_ave['lambda'] * pt_ave['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * pt_ave['sub_wgt'][ib]

                # Unify subgroup flux
                pt_ave['sub_flx'] = sub_flux
                pts.append(pt_ave)
                unify_sub_wgt(pts, has_resfis)
                sub_flux = pt_ave['sub_flx']
                pts = pts[0:self._n_res_reg]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pts=pts)

    def solve_adjust_numdens(self):
        # Solve problem at effective temperature
        if self.pin_cell_ave is None:
            raise Exception('pin_cell_ave should be given')
        ave_solver = ResonancePinSubgroup()
        ave_solver.pin_cell = self.pin_cell_ave
        ave_solver.pin_solver = self.pin_solver
        ave_solver.micro_lib = self.micro_lib
        ave_solver.use_pseudo_lib = self.use_pseudo_lib
        ave_solver.cross_sections = self.cross_sections
        ave_solver.first_calc_g = self.first_calc_g
        ave_solver.last_calc_g = self.last_calc_g
        ave_solver.solve_onenuc_onetemp()

        if self._pin_cell.ave_temp is None:
            raise Exception('ave_temp of PinCell should be given')

        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Initialize pseudo library at average temperature
        res_nuc = None
        res_nuc_temps = []
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_nuc_temps.append(mat.temperature)
                    break
        self._init_plib_one_nuc(res_nuc, 1.0, self._pin_cell.ave_temp)

        # Whether has resonance fission
        has_resfis = self._micro_lib.has_resfis(res_nuc)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get subgroup parameters at average temperature
            pt_ave = self._pseudo_lib.get_subp_one_nuc(ig, res_nuc)
            n_band = pt_ave['n_band']

            print(ig, n_band)
            if n_band > 1:
                # Compute subgroup levels
                pts = comp_sub_level(ig, pt_ave, self._micro_lib,
                                     ave_solver.resnuc_xs, res_nuc,
                                     res_nuc_temps,
                                     self._pin_cell.ave_temp, has_resfis)

                sub_flux = np.zeros((n_band, n_region))
                for ib in range(n_band):
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        if ireg < len(pts):
                            subp = pts[ireg]
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_sca'][ib] * \
                                    (1.0 - subp['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    subp['lambda'] * subp['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * subp['sub_wgt'][ib]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pts=pts)

    def solve_correlation(self):
        if self._pin_cell.ave_temp is None:
            raise Exception('ave_temp of PinCell should be given')

        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Initialize pseudo library
        res_nuc_temps = []
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_nuc_dens = mat.densities[inuc]
                    res_nuc_temps.append(mat.temperature)
                    break
        self._init_plib_temp_as_nuc(res_nuc, res_nuc_dens, res_nuc_temps,
                                    self._pin_cell.ave_temp)

        # Whether has resonance fission
        has_resfis = self._micro_lib.has_resfis(res_nuc)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get subgroup parameters at different temperatures
            pts = []
            for itemp, temp in enumerate(res_nuc_temps):
                plib = self._pseudo_libs[itemp]
                pts.append(plib.get_subp_one_nuc(ig, res_nuc))

            # Unify subgroup parameters
            n_band = unify_sub_wgt(pts, has_resfis)

            print(ig, n_band)
            if n_band > 1:
                sub_flux = np.zeros((n_band, n_region))
                for ib in range(n_band):
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        if ireg < len(pts):
                            subp = pts[ireg]
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_sca'][ib] * \
                                    (1.0 - subp['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    subp['lambda'] * subp['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * subp['sub_wgt'][ib]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pts=pts)

    def solve_adjust_sub_level(self):
        if self._pin_cell.ave_temp is None:
            raise Exception('ave_temp of PinCell should be given')

        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Initialize pseudo library
        res_nuc_temps = []
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_nuc_dens = mat.densities[inuc]
                    res_nuc_temps.append(mat.temperature)
                    break
        self._init_plib_temp_as_nuc(res_nuc, res_nuc_dens, res_nuc_temps,
                                    self._pin_cell.ave_temp)

        # Whether has resonance fission
        has_resfis = self._micro_lib.has_resfis(res_nuc)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get resonance tables at different temperatures
            rxs = []
            for itemp in range(len(res_nuc_temps)+1):
                rxs.append(self._pseudo_libs[itemp].get_res_tbl(ig, res_nuc))

            # Get subgroup parameters at different temperatures
            pts = []
            for itemp in range(len(res_nuc_temps)+1):
                plib = self._pseudo_libs[itemp]
                pts.append(plib.get_subp_one_nuc(ig, res_nuc))

            # Adjust subgroup parameters
            n_band = adjust_sub_level(pts, rxs, has_resfis)

            print(ig, n_band)
            if n_band > 1:
                sub_flux = np.zeros((n_band, n_region))
                for ib in range(n_band):
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        if ireg < len(pts):
                            subp = pts[ireg]
                            # print subp
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_sca'][ib] * \
                                    (1.0 - subp['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    subp['lambda'] * subp['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * subp['sub_wgt'][ib]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pts=pts)

    def solve_onenuc_onetemp(self):
        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._count_n_res_reg()

        # Initialize subgroup macro xs
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Get the resonant nuclide and resonant material
        for mat in self._pin_cell.materials:
            for inuc, nuc in enumerate(mat.nuclides):
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_mat = mat
                    res_nuc_dens = mat.densities[inuc]
                    res_mat_temp = mat.temperature
                    break

        # Initialize pseudo library if needed
        if self._use_pseudo_lib:
            self._init_plib_one_nuc(res_nuc, res_nuc_dens, res_mat_temp)

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        if self._first_calc_g is None:
            first_calc_g = self._micro_lib.first_res
        else:
            first_calc_g = self._first_calc_g
        if self._last_calc_g is None:
            last_calc_g = self._micro_lib.last_res
        else:
            last_calc_g = self._last_calc_g
        for ig in range(first_calc_g, last_calc_g):
            # Get subgroup parameters
            if self._use_pseudo_lib:
                subp = self._pseudo_lib.get_subp_one_nuc(ig, res_nuc)
            else:
                subp = self._micro_lib.get_subp(res_nuc, res_mat.temperature,
                                                ig)

            print(ig, subp['n_band'])
            if subp['n_band'] > 1:
                sub_flux = np.zeros((subp['n_band'], n_region))
                for ib in range(subp['n_band']):
                    # Calculate subgroup macro xs
                    mac_tot[:] = 0.0
                    mac_sca[:] = 0.0
                    mac_src[:] = 0.0
                    for ireg in range(n_region):
                        imat = self._pin_cell.mat_fill[ireg]
                        mat = self._pin_cell.materials[imat]
                        for inuc, nuc in enumerate(mat.nuclides):
                            if nuc == res_nuc:
                                mac_tot[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += mat.densities[inuc] \
                                    * subp['sub_sca'][ib] * \
                                    (1.0 - subp['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    subp['lambda'] * subp['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += mat.densities[inuc] * xs_typ['total']
                                mac_sca[ireg] \
                                    += mat.densities[inuc] * xs_typ['scatter']\
                                    * (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += mat.densities[inuc] * \
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux \
                        * subp['sub_wgt'][ib]

                # Condense self-shielded xs
                self._condense_self_shielded_xs(
                    ig, res_nuc, sub_flux, self._pin_solver.vols, pt=subp)

    def _init_plib_one_nuc_multi_temp(self, nuclide, density, temperatures,
                                      ave_temp):
        if self._cross_sections is None:
            raise Exception('cross_sections should be given')

        # Input for library of pseudo nuclide
        self._pseudo_lib = LibraryPseudo()
        self._pseudo_lib.nuclides = [nuclide]
        self._pseudo_lib.densities = [density]
        self._pseudo_lib.temperature = ave_temp
        self._pseudo_lib.temps = temperatures
        self._pseudo_lib.cross_sections = self._cross_sections
        self._pseudo_lib.mglib = self._micro_lib

        # Make the library
        self._pseudo_lib.make()

    def _init_plib_temp_as_nuc(self, nuclide, density, temperatures,
                               ave_temp):
        if self._cross_sections is None:
            raise Exception('cross_sections should be given')

        self._pseudo_libs = []
        temps = deepcopy(temperatures)
        temps.append(ave_temp)
        for temp in temps:
            plib = LibraryPseudo()
            plib.nuclides = [nuclide]
            plib.densities = [density]
            plib.temperature = temp
            plib.cross_sections = self._cross_sections
            plib.mglib = self._micro_lib
            plib.make()
            self._pseudo_libs.append(plib)

    def _init_plib_one_nuc(self, nuclide, density, temperature):
        if self._cross_sections is None:
            raise Exception('cross_sections should be given')

        # Input for library of pseudo nuclide
        self._pseudo_lib = LibraryPseudo()
        self._pseudo_lib.nuclides = [nuclide]
        self._pseudo_lib.densities = [density]
        self._pseudo_lib.temperature = temperature
        self._pseudo_lib.cross_sections = self._cross_sections
        self._pseudo_lib.mglib = self._micro_lib

        # Make the library
        self._pseudo_lib.make()

    def _init_self_shielded_xs(self):
        # Initialize (pin averaged) self_shielded xs at typical dilution
        n_res = self._micro_lib.last_res - self._micro_lib.first_res
        for nuc in self._resnuc_xs:
            self._resnuc_xs[nuc]['xs_abs'] \
                = np.zeros((n_res, self._n_res_reg+1))
            self._resnuc_xs[nuc]['xs_sca'] \
                = np.zeros((n_res, self._n_res_reg+1))
            if self._micro_lib.has_resfis(nuc):
                self._resnuc_xs[nuc]['xs_nfi'] \
                    = np.zeros((n_res, self._n_res_reg+1))
            for ireg in range(self._n_res_reg):
                imat = self._pin_cell.mat_fill[ireg]
                mat = self._pin_cell.materials[imat]
                for ig in range(self._micro_lib.first_res,
                                self._micro_lib.last_res):
                    jg = ig - self._micro_lib.first_res
                    xs_typ = self._micro_lib.get_typical_xs(
                        nuc, mat.temperature, ig, 'scatter', 'absorb',
                        'nufis')
                    self._resnuc_xs[nuc]['xs_abs'][jg, ireg] \
                        = xs_typ['absorb']
                    self._resnuc_xs[nuc]['xs_sca'][jg, ireg] \
                        = xs_typ['scatter']
                    if 'xs_nfi' in self._resnuc_xs[nuc]:
                        self._resnuc_xs[nuc]['xs_nfi'][jg, ireg] \
                            = xs_typ['nufis']
            for ig in range(self._micro_lib.first_res,
                            self._micro_lib.last_res):
                jg = ig - self._micro_lib.first_res
                self._resnuc_xs[nuc]['xs_abs'][jg, self._n_res_reg] \
                    = self._resnuc_xs[nuc]['xs_abs'][jg, 0]
                self._resnuc_xs[nuc]['xs_sca'][jg, self._n_res_reg] \
                    = self._resnuc_xs[nuc]['xs_sca'][jg, 0]
                if 'xs_nfi' in self._resnuc_xs[nuc]:
                    self._resnuc_xs[nuc]['xs_nfi'][jg, self._n_res_reg] \
                        = self._resnuc_xs[nuc]['xs_nfi'][jg, 0]

    def _condense_self_shielded_xs(self, ig, nuc, sub_flux, vols, pt=None,
                                   pts=None):
        jg = ig - self._micro_lib.first_res
        n = self._n_res_reg
        n_band = sub_flux.shape[0]
        for ib in range(n_band):
            sub_flux[ib, :] *= vols[:]
        flux = np.sum(sub_flux, 0)
        sum_flux = np.sum(flux[0:n])

        # Condense subgroup xs to self-shielded xs
        for ireg in range(n):
            if pt is not None:
                subp = pt
            elif pts is not None:
                subp = pts[ireg]
            else:
                raise Exception('pt or pts should be given')
            self._resnuc_xs[nuc]['xs_abs'][jg, ireg] \
                = np.sum(subp['sub_abs'] * sub_flux[:, ireg]) \
                / flux[ireg]
            self._resnuc_xs[nuc]['xs_sca'][jg, ireg] \
                = np.sum(subp['sub_sca'] * sub_flux[:, ireg]) \
                / flux[ireg]
            if 'xs_nfi' in self._resnuc_xs[nuc]:
                self._resnuc_xs[nuc]['xs_nfi'][jg, ireg] \
                    = np.sum(subp['sub_nfi'] * sub_flux[:, ireg]) \
                    / flux[ireg]

        # Condense spatial dependent self-shielded xs to pin averaged
        # self-shielded xs
        xs = self._resnuc_xs[nuc]['xs_abs']
        xs[jg, n] = np.sum(xs[jg, 0:n] * flux[0:n]) / sum_flux
        xs = self._resnuc_xs[nuc]['xs_sca']
        xs[jg, n] = np.sum(xs[jg, 0:n] * flux[0:n]) / sum_flux
        self._resnuc_xs[nuc]['xs_tot'] = self._resnuc_xs[nuc]['xs_abs'] \
            + self._resnuc_xs[nuc]['xs_sca']
        if 'xs_nfi' in self._resnuc_xs[nuc]:
            xs = self._resnuc_xs[nuc]['xs_nfi']
            xs[jg, n] = np.sum(xs[jg, 0:n] * flux[0:n]) / sum_flux

    def print_self_shielded_xs(self, to_screen=True, to_h5=None):
        n_res = self._micro_lib.last_res - self._micro_lib.first_res
        n = self._n_res_reg
        if to_screen:
            for nuc in self._resnuc_xs:
                print('nuclide', nuc)
                print('pin averaged')
                for ig in range(n_res):
                    print("%4i %f" %
                          (ig, self._resnuc_xs[nuc]['xs_abs'][ig, n]))

        if to_h5 is not None:
            f = h5py.File(to_h5, 'w')
            for nuc in self._resnuc_xs:
                f.create_group(nuc)
                f[nuc].create_group('regions')
                f[nuc]['regions']['xs_abs'] \
                    = self._resnuc_xs[nuc]['xs_abs'][:, :self._n_res_reg]
                f[nuc].create_group('average')
                f[nuc]['average']['xs_abs'] \
                    = self._resnuc_xs[nuc]['xs_abs'][:, self._n_res_reg]
            f.close()

    def _count_n_res_reg(self):
        self._n_res_reg = 0
        for ireg in range(len(self._pin_cell.mat_fill)):
            imat = self._pin_cell.mat_fill[ireg]
            mat = self._pin_cell.materials[imat]
            has_res = False
            for nuc in mat.nuclides:
                if self._micro_lib.has_res(nuc):
                    has_res = True
                    break
            if not has_res:
                self._n_res_reg = ireg
                break

    @property
    def pin_cell(self):
        return self._pin_cell

    @pin_cell.setter
    def pin_cell(self, pin_cell):
        self._pin_cell = pin_cell

    @property
    def pin_solver(self):
        return self._pin_solver

    @pin_solver.setter
    def pin_solver(self, pin_solver):
        self._pin_solver = pin_solver

    @property
    def micro_lib(self):
        return self._micro_lib

    @micro_lib.setter
    def micro_lib(self, micro_lib):
        self._micro_lib = micro_lib

    @property
    def use_pseudo_lib(self):
        return self._use_pseudo_lib

    @use_pseudo_lib.setter
    def use_pseudo_lib(self, use_pseudo_lib):
        self._use_pseudo_lib = use_pseudo_lib

    @property
    def cross_sections(self):
        return self._cross_sections

    @cross_sections.setter
    def cross_sections(self, cross_sections):
        self._cross_sections = cross_sections

    @property
    def first_calc_g(self):
        return self._first_calc_g

    @first_calc_g.setter
    def first_calc_g(self, first_calc_g):
        self._first_calc_g = first_calc_g

    @property
    def last_calc_g(self):
        return self._last_calc_g

    @last_calc_g.setter
    def last_calc_g(self, last_calc_g):
        self._last_calc_g = last_calc_g

    @property
    def pin_cell_ave(self):
        return self._pin_cell_ave

    @pin_cell_ave.setter
    def pin_cell_ave(self, pin_cell_ave):
        self._pin_cell_ave = pin_cell_ave

    @property
    def resnuc_xs(self):
        return self._resnuc_xs

    @resnuc_xs.setter
    def resnuc_xs(self, resnuc_xs):
        self._resnuc_xs = resnuc_xs

    @property
    def n_res_reg(self):
        return self._n_res_reg

    @n_res_reg.setter
    def n_res_reg(self, n_res_reg):
        self._n_res_reg = n_res_reg


class PinFixSolver(object):

    def __init__(self):
        # Geometry definition
        self._pin_type = None
        self._radii = None
        self._pitch = None
        self._n_ring_fuel = 10
        self._n_ring_fuel_mat = None
        self._n_sector_fuel = 1
        self._n_sector_mod = 8

        # Cross sections definition
        self._xs_tot = None
        self._xs_sca = None
        self._source = None

        # Volume averaged material region flux
        self._flux = None
        self._vols = None

        # Moc solver
        self._moc_opts = _openmoc_opts
        self._moc_setup_solver_done = False
        self._moc_solver = None
        self._moc_materials = None
        self._moc_fsr2mat = None

        self._is_init = False

    def set_pin_geometry(self, pin_cell):
        self._pin_type = pin_cell.pin_type
        self._radii = pin_cell.radii
        self._pitch = pin_cell.pitch

    def set_pin_xs(self, xs_tot=None, xs_sca=None, source=None):
        if self._pin_type == PINCELLBOX:
            self._moc_set_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
        else:
            raise Exception('not support PINCELLCYL')

    def _moc_set_xs(self, xs_tot=None, xs_sca=None, source=None):
        # Set up MOC solver
        if not self._moc_setup_solver_done:
            self._moc_setup_solver()

        # Get the user xs or xs of the object
        if xs_tot is None:
            _xs_tot = self._xs_tot
        else:
            _xs_tot = xs_tot
            self._xs_tot = xs_tot
        if xs_sca is None:
            _xs_sca = self._xs_sca
        else:
            _xs_sca = xs_sca
            self._xs_sca = xs_sca
        if source is None:
            _source = self._source
        else:
            _source = source
            self._source = source

        for imat, material in enumerate(self._moc_materials):
            # Set number of energy groups
            material.setNumEnergyGroups(1)

            # Set total xs
            material.setSigmaT(np.array([_xs_tot[imat]]))

            # Set scatter xs
            material.setSigmaS(np.array([_xs_sca[imat]]))

            # Set chi and nu fission to be zero
            material.setNuSigmaF(np.zeros(1))
            material.setChi(np.zeros(1))

        # Set source
        geometry = self._moc_solver.getGeometry()
        n_fsr = geometry.getNumFSRs()
        for ifsr in range(n_fsr):
            imat = self._moc_fsr2mat[ifsr]
            self._moc_solver.setFixedSourceByFSR(ifsr, 1, _source[imat])

    def _moc_setup_solver(self):
        # Creating surfaces
        n_radii = len(self._radii)
        zcylinders = []
        for i, r in enumerate(self._radii):
            zcylinders.append(openmoc.ZCylinder(
                x=0.0, y=0.0, radius=r, name='zcylinder%i' % (i)))
        half_pitch = self._pitch / 2.0
        left = openmoc.XPlane(x=-half_pitch, name='left')
        right = openmoc.XPlane(x=half_pitch, name='right')
        top = openmoc.YPlane(y=half_pitch, name='top')
        bottom = openmoc.YPlane(y=-half_pitch, name='bottom')
        left.setBoundaryType(openmoc.REFLECTIVE)
        right.setBoundaryType(openmoc.REFLECTIVE)
        top.setBoundaryType(openmoc.REFLECTIVE)
        bottom.setBoundaryType(openmoc.REFLECTIVE)

        # Creating empty materials for each fuel ring and moderator region
        self._moc_materials = []
        for i in range(n_radii+1):
            self._moc_materials.append(openmoc.Material(id=i+1))

        # Creating cells and universe
        self._n_ring_fuel_mat = int(ceil(float(self._n_ring_fuel) / n_radii))
        root_universe = openmoc.Universe(name='root universe')
        for i in range(n_radii):
            cell = openmoc.Cell(id=i+1)
            cell.setNumRings(self._n_ring_fuel_mat)
            cell.setNumSectors(self._n_sector_fuel)
            cell.setFill(self._moc_materials[i])
            cell.addSurface(halfspace=-1, surface=zcylinders[i])
            if i > 0:
                cell.addSurface(halfspace=1, surface=zcylinders[i-1])
            root_universe.addCell(cell)
        cell = openmoc.Cell(id=n_radii+1)
        cell.setNumSectors(self._n_sector_mod)
        cell.setFill(self._moc_materials[n_radii])
        cell.addSurface(halfspace=1, surface=zcylinders[n_radii-1])
        cell.addSurface(halfspace=+1, surface=left)
        cell.addSurface(halfspace=-1, surface=right)
        cell.addSurface(halfspace=+1, surface=bottom)
        cell.addSurface(halfspace=-1, surface=top)
        root_universe.addCell(cell)

        # Creating the geometry
        geometry = openmoc.Geometry()
        geometry.setRootUniverse(root_universe)

        # Creating the track
        track = openmoc.TrackGenerator(geometry, self._moc_opts.num_azim,
                                       self._moc_opts.azim_spacing)
        track.setNumThreads(self._moc_opts.num_omp_threads)
        track.generateTracks()

        # Set up solver
        self._moc_solver = openmoc.CPUSolver(track)
        self._moc_solver.useExponentialIntrinsic()
        self._moc_solver.setNumThreads(self._moc_opts.num_omp_threads)
        self._moc_solver.setConvergenceThreshold(self._moc_opts.tolerance)

        # Map fsr to material index
        n_fsr = geometry.getNumFSRs()
        self._moc_fsr2mat = np.zeros(n_fsr, dtype=int)
        for ifsr in range(n_fsr):
            self._moc_fsr2mat[ifsr] = geometry.findFSRMaterial(ifsr).getId()-1

        # Initialize the fluxes
        self._flux = np.zeros(n_radii+1)

        self._moc_setup_solver_done = True

    def _solve_moc(self):
        geometry = self._moc_solver.getGeometry()
        n_fsr = geometry.getNumFSRs()

        # Solver the fixed source problem with scattering iteration
        self._moc_solver.iterateScattSource()
        fsr_fluxes = get_scalar_fluxes(self._moc_solver)

        # Get the volumes
        self._vols = np.zeros(len(self._radii)+1)
        for ifsr in range(n_fsr):
            imat = self._moc_fsr2mat[ifsr]
            self._vols[imat] += self._moc_solver.getFSRVolume(ifsr)

        # Compute the material averaged fluxes
        self._flux[:] = 0.0
        for ifsr in range(n_fsr):
            imat = self._moc_fsr2mat[ifsr]
            vol = self._moc_solver.getFSRVolume(ifsr)
            self._flux[imat] += fsr_fluxes[ifsr, 0] * vol
        self._flux[:] /= self._vols[:]

    def solve(self):
        if self._pin_type == PINCELLBOX:
            self._solve_moc()
        else:
            raise Exception('not support PINCELLCYL')

    @property
    def pin_type(self):
        return self._pin_type

    @pin_type.setter
    def pin_type(self, pin_type):
        self._pin_type = pin_type

    @property
    def radii(self):
        return self._radii

    @radii.setter
    def radii(self, radii):
        self._radii = radii

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = pitch

    @property
    def xs_tot(self):
        return self._xs_tot

    @xs_tot.setter
    def xs_tot(self, xs_tot):
        self._xs_tot = xs_tot

    @property
    def xs_sca(self):
        return self._xs_sca

    @xs_sca.setter
    def xs_sca(self, xs_sca):
        self._xs_sca = xs_sca

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, flux):
        self._flux = flux

    @property
    def moc_opts(self):
        return self._moc_opts

    @moc_opts.setter
    def moc_opts(self, moc_opts):
        self._moc_opts = moc_opts

    @property
    def moc_setup_solver_done(self):
        return self._moc_setup_solver_done

    @moc_setup_solver_done.setter
    def moc_setup_solver_done(self, moc_setup_solver_done):
        self._moc_setup_solver_done = moc_setup_solver_done

    @property
    def moc_solver(self):
        return self._moc_solver

    @moc_solver.setter
    def moc_solver(self, moc_solver):
        self._moc_solver = moc_solver

    @property
    def moc_materials(self):
        return self._moc_materials

    @moc_materials.setter
    def moc_materials(self, moc_materials):
        self._moc_materials = moc_materials

    @property
    def n_ring_fuel(self):
        return self._n_ring_fuel

    @n_ring_fuel.setter
    def n_ring_fuel(self, n_ring_fuel):
        self._n_ring_fuel = n_ring_fuel

    @property
    def n_sector_fuel(self):
        return self._n_sector_fuel

    @n_sector_fuel.setter
    def n_sector_fuel(self, n_sector_fuel):
        self._n_sector_fuel = n_sector_fuel

    @property
    def n_sector_mod(self):
        return self._n_sector_mod

    @n_sector_mod.setter
    def n_sector_mod(self, n_sector_mod):
        self._n_sector_mod = n_sector_mod

    @property
    def vols(self):
        return self._vols

    @vols.setter
    def vols(self, vols):
        self._vols = vols
