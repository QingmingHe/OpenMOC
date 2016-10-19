#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from library_micro import LibraryMicro
from time import clock
from pin_solver import PinFixSolver, PINCELLBOX


class Nuclide(object):
    def __init__(self):
        self.name = None
        self.density = None


class Material(object):
    def __init__(self):
        self.temperature = None
        self.nuclides = None  # Nuclide list
        self.name = None  # Material name


class PinCell(object):
    def __init__(self):
        # Geometry definition
        self._pin_type = None
        self._radii = None
        self._pitch = None

        # Material definition (Matieral list)
        self._materials = None

        # Materials in material regions
        self._mat_fill = None

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


class ResonancePinSubgroup(object):

    def __init__(self):
        # Given by user
        self._pin_cell = None
        self._pin_solver = None
        self._micro_lib = None

        self._resnuc_xs = None
        self._n_res_reg = None

    def _solve_onenuc_onetemp(self):
        n_res = self._micro_lib.last_res - self._micro_lib.first_res
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Get the first resonant nuclide and resonant material
        res_nuc = None
        res_mat = None
        for mat in self._pin_cell.materials:
            for nuc in mat.nuclides:
                if self._micro_lib.has_res(nuc.name):
                    res_nuc = nuc.name
                    res_mat = mat
                    break

        # Initialize self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._resnuc_xs[res_nuc]['xs_abs'] \
            = np.zeros((n_res, self._n_res_reg))
        self._resnuc_xs[res_nuc]['xs_sca'] \
            = np.zeros((n_res, self._n_res_reg))
        if self._micro_lib.has_resfis(res_nuc):
            self._resnuc_xs[res_nuc]['xs_nfi'] \
                = np.zeros((n_res, self._n_res_reg))

        # for ig in range(22, 24):
        for ig in range(self._micro_lib.first_res, self._micro_lib.last_res):
            jg = ig - self._micro_lib.first_res
            # Get subgroup parameters
            subp = self._micro_lib.get_subp(res_nuc, res_mat.temperature, ig)

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
                        for nuc in mat.nuclides:
                            if nuc.name == res_nuc:
                                mac_tot[ireg] \
                                    += nuc.density * subp['sub_tot'][ib]
                                mac_sca[ireg] \
                                    += nuc.density * subp['sub_sca'][ib] * \
                                    (1.0 - subp['lambda'])
                                mac_src[ireg] \
                                    += nuc.density * subp['sub_wgt'][ib] * \
                                    subp['lambda'] * subp['potential']
                            else:
                                xs_typ = self._micro_lib.get_typical_xs(
                                    nuc.name, mat.temperature, ig, 'total',
                                    'scatter', 'potential', 'lambda')
                                mac_tot[ireg] \
                                    += nuc.density * xs_typ['total']
                                mac_sca[ireg] \
                                    += nuc.density * xs_typ['scatter'] * \
                                    (1.0 - xs_typ['lambda'])
                                mac_src[ireg] \
                                    += nuc.density * subp['sub_wgt'][ib] *\
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux[:]

                # Condense self-shielded xs
                for ireg in range(self._n_res_reg):
                    flux = np.sum(sub_flux[:, ireg])
                    self._resnuc_xs[res_nuc]['xs_abs'][jg, ireg] \
                        = np.sum(subp['sub_abs'] * sub_flux[:, ireg]) / flux
                    self._resnuc_xs[res_nuc]['xs_sca'][jg, ireg] \
                        = np.sum(subp['sub_sca'] * sub_flux[:, ireg]) / flux
                    if 'xs_nfi' in self._resnuc_xs[res_nuc]:
                        self._resnuc_xs[res_nuc]['xs_nfi'][jg, ireg] \
                            = np.sum(subp['sub_nfi'] * sub_flux[:, ireg]) \
                            / flux

            else:
                # Get xs at typical dilution for resonant nuclide
                for ireg in range(self._n_res_reg):
                    imat = self._pin_cell.mat_fill[ireg]
                    mat = self._pin_cell.materials[imat]
                    xs_typ = self._micro_lib.get_typical_xs(
                        res_nuc, mat.temperature, ig, 'scatter', 'absorb',
                        'nufis')
                    self._resnuc_xs[res_nuc]['xs_abs'][jg, ireg] \
                        = xs_typ['absorb']
                    self._resnuc_xs[res_nuc]['xs_sca'][jg, ireg] \
                        = xs_typ['scatter']
                    if 'xs_nfi' in self._resnuc_xs[res_nuc]:
                        self._resnuc_xs[res_nuc]['xs_nfi'][jg, ireg] \
                            = xs_typ['nufis']

            print jg, self._resnuc_xs[res_nuc]['xs_abs'][jg, ireg]

    def solve(self):
        # Set pin geometry
        self._pin_solver.set_pin_geometry(self._pin_cell)

        # Count number of resonant region
        self._n_res_reg = 0
        for ireg in range(len(self._pin_cell.mat_fill)):
            imat = self._pin_cell.mat_fill[ireg]
            mat = self._pin_cell.materials[imat]
            has_res = False
            for nuc in mat.nuclides:
                if self._micro_lib.has_res(nuc.name):
                    has_res = True
                    break
            if not has_res:
                self._n_res_reg = ireg
                break

        # Solver one nuclide on temperature
        self._solve_onenuc_onetemp()

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
    def resnuc_xs(self):
        return self._resnuc_xs

    @resnuc_xs.setter
    def resnuc_xs(self, resnuc_xs):
        self._resnuc_xs = resnuc_xs


def test_subgroup():
    # Define nuclides
    h1 = Nuclide()
    h1.name = 'H1'
    h1.density = 0.0662188
    u238 = Nuclide()
    u238.name = 'U238'
    u238.density = 2.21546e-2
    # Define materials
    fuel = Material()
    fuel.temperature = 293.6
    fuel.nuclides = [u238]
    fuel.name = 'fuel'
    mod = Material()
    mod.temperature = 293.6
    mod.nuclides = [h1]
    mod.name = 'mod'
    # Load micro library
    import os
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmc/openmc/micromgxs',
                         'jeff-3.2-wims69e.h5')
    lib = LibraryMicro()
    lib.load_from_h5(fname)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.radii = [0.4095]
    pin.pitch = 1.26
    pin.materials = [fuel, mod]
    pin.mat_fill = [0, 1]
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.micro_lib = lib
    sub.solve()

if __name__ == '__main__':
    test_subgroup()
