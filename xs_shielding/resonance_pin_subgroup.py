#!/usr/bin/env python
# -*- coding: utf-8 -*-
import openmoc
from openmoc.process import get_scalar_fluxes
import numpy as np
from math import ceil
from library_micro import LibraryMicro
from time import clock

PINCELLBOX = 1
PINCELLCYL = 2
openmoc.log.set_log_level('TITLE')
_openmoc_opts = openmoc.options.Options()


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

        # Creating empty materials
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

        # Compute the material averaged fluxes
        self._flux[:] = 0.0
        n_mat = len(self._moc_materials)
        vols = np.zeros(n_mat)
        for ifsr in range(n_fsr):
            imat = self._moc_fsr2mat[ifsr]
            vol = self._moc_solver.getFSRVolume(ifsr)
            vols[imat] += vol
            self._flux[imat] += fsr_fluxes[ifsr, 0] * vol
        self._flux[:] /= vols[:]

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


def test_pinsolver():
    # Test a simple pin problem
    p = PinFixSolver()
    p.pin_type = PINCELLBOX
    p.radii = [0.4095]
    p.pitch = 1.26

    # Test 1
    time0 = clock()
    xs_tot = [1.53097167e+03, 1.35399647e+00]
    xs_sca = [612.38866698, 0.]
    source = [2.13863960e-05, 4.27554870e-04]
    p.set_pin_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
    p.solve()
    print('time for test 1:')
    print(clock() - time0)

    # Test 2
    time0 = clock()
    xs_tot = [0.02887092, 1.35346599]
    xs_sca = [0.01154837, 0.]
    source = [0.05097604, 1.26152386]
    p.set_pin_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
    p.solve()
    print('time for test 2:')
    print(clock() - time0)

    # Test 3. Time is longer than test 1
    time0 = clock()
    xs_tot = [1.53097167e+03, 1.35399647e+00]
    xs_sca = [612.38866698, 0.]
    source = [2.13863960e-05, 4.27554870e-04]
    p.set_pin_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
    p.solve()
    print('time for 2nd run of test 1:')
    print(clock() - time0)
    print('why longer than 1st run of test 1')

if __name__ == '__main__':
    # test_subgroup()
    test_pinsolver()
