# -*- coding: utf-8 -*-
import numpy as np
from time import clock
import openmoc
from openmoc.process import get_scalar_fluxes
from math import ceil

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
    test_pinsolver()
