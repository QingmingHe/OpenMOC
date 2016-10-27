#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import openmoc
from openmoc.process import get_scalar_fluxes
from math import ceil
from library_micro import LibraryMicro
from time import clock
import h5py

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
                 mat_fill=None):
        # Geometry definition
        self._pin_type = pin_type
        self._radii = radii
        self._pitch = pitch

        # Material definition (Matieral list)
        self._materials = materials

        # Materials in material regions
        self._mat_fill = mat_fill

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

    def __init__(self, pin_cell=None, pin_solver=None, micro_lib=None):
        # Given by user
        self._pin_cell = None
        self._pin_solver = None
        self._micro_lib = None

        self._resnuc_xs = None
        self._n_res_reg = None

    def _solve_adjust_sub_wgt(self):
        pass

    def _solve_onenuc_onetemp(self):
        n_region = len(self._pin_cell.mat_fill)
        mac_tot = np.zeros(n_region)
        mac_sca = np.zeros(n_region)
        mac_src = np.zeros(n_region)

        # Get the first resonant nuclide and resonant material
        res_nuc = None
        res_mat = None
        for mat in self._pin_cell.materials:
            for nuc in mat.nuclides:
                if self._micro_lib.has_res(nuc):
                    res_nuc = nuc
                    res_mat = mat
                    break

        # Initialize self-shielded xs. Xs for last region is pin averaged
        # self-shielded xs
        self._resnuc_xs = {}
        self._resnuc_xs[res_nuc] = {}
        self._init_self_shielded_xs()

        for ig in range(self._micro_lib.first_res, self._micro_lib.last_res):
            # Get subgroup parameters
            subp = self._micro_lib.get_subp(res_nuc, res_mat.temperature, ig)

            print ig, subp['n_band']
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
                                    subp['sub_wgt'][ib] * \
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
                                    subp['sub_wgt'][ib] *\
                                    xs_typ['lambda'] * xs_typ['potential']

                    # Solve subgroup fixed source equation and get volume
                    # integrated flux
                    self._pin_solver.set_pin_xs(xs_tot=mac_tot, xs_sca=mac_sca,
                                                source=mac_src)
                    self._pin_solver.solve()
                    sub_flux[ib, :] = self._pin_solver.flux

                # Condense self-shielded xs
                self._condense_self_shielded_xs(ig, subp, res_nuc, sub_flux,
                                                self._pin_solver.vols)

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

    def _condense_self_shielded_xs(self, ig, subp, nuc, sub_flux, vols):
        jg = ig - self._micro_lib.first_res
        n = self._n_res_reg
        n_band = sub_flux.shape[0]
        for ib in range(n_band):
            sub_flux[ib, :] *= vols[:]
        flux = np.sum(sub_flux, 0)
        sum_flux = np.sum(flux[0:n])

        # Condense subgroup xs to self-shielded xs
        for ireg in range(n):
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
        if 'xs_nfi' in self._resnuc_xs[nuc]:
            xs = self._resnuc_xs[nuc]['xs_nfi']
            xs[jg, n] = np.sum(xs[jg, 0:n] * flux[0:n]) / sum_flux

    def print_self_shielded_xs(self, to_screen=True, to_h5=None):
        n_res = self._micro_lib.last_res - self._micro_lib.first_res
        n = self._n_res_reg
        if to_screen:
            for nuc in self._resnuc_xs:
                print('nuclide', nuc)
                for ireg in range(n):
                    print('region', ireg)
                    for ig in range(n_res):
                        print(ig, self._resnuc_xs[nuc]['xs_abs'][ig, ireg])
                print('pin averaged')
                for ig in range(n_res):
                    print(ig, self._resnuc_xs[nuc]['xs_abs'][ig, n])

        if to_h5 is not None:
            f = h5py.File(to_h5, 'w')
            for nuc in self._resnuc_xs:
                f.create_group(nuc)
                for ireg in range(self._n_res_reg):
                    reg = 'region%i' % ireg
                    f[nuc].create_group(reg)
                    f[nuc][reg]['xs_abs'] \
                        = self._resnuc_xs[nuc]['xs_abs'][:, ireg]
                f[nuc].create_group('average')
                f[nuc]['average']['xs_abs'] \
                    = self._resnuc_xs[nuc]['xs_abs'][:, self._n_res_reg]
            f.close()

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
                if self._micro_lib.has_res(nuc):
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


def test_subgroup():
    # Define materials
    fuel = Material(temperature=293.6, nuclides=['U238'],
                    densities=[2.21546e-2], name='fuel')
    mod = Material(temperature=293.6, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    import os
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmc/openmc/micromgxs',
                         'jeff-3.2-wims69e-25m.h5')
    lib = LibraryMicro()
    lib.load_from_h5(fname)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel, mod]
    pin.radii = [
        0.129653384067,
        0.183357574155,
        0.224566248577,
        0.259306768134,
        0.289913780286,
        0.317584634389,
        0.343030610879,
        0.36671514831,
        0.388960152201,
        0.41
    ]
    pin.mat_fill = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.micro_lib = lib
    sub.solve()
    sub.print_self_shielded_xs(to_h5='simple-pin.h5', to_screen=False)

if __name__ == '__main__':
    test_subgroup()
