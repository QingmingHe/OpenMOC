#!/usr/bin/env python
# -*- coding: utf-8 -*-
import openmoc
import openmc
import h5py
from library_micro import LibraryMicro
import os
from scipy.integrate import quad
from math import exp, sinh, pi, sqrt
import numpy as np

###############################################################################
#                          Main Simulation Parameters
###############################################################################

opts = openmoc.options.Options()

openmoc.log.set_log_level('NORMAL')

name = 'simple-pin-975'
fname = '%s-macro.h5' % name
spfile = 'one_nuc_multi_temp/dist/statepoint.2000.h5'
with h5py.File(fname) as f:
    ng = f.attrs['# groups']
    n_region = len(f['material'])

###############################################################################
#                            Creating Materials
###############################################################################

openmoc.log.py_printf('NORMAL', 'Importing materials data from HDF5...')

materials = openmoc.materialize.load_from_hdf5(fname, './')

###############################################################################
#                            Creating Surfaces
###############################################################################
openmoc.log.py_printf('NORMAL', 'Creating surfaces...')

n_radii = 10
radii = [
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
R = radii[-1]
pitch = 1.26
half_pitch = pitch / 2.0
zcylinders = []
vols = np.zeros(n_region) + pi * radii[0] ** 2
vols[-1] = pitch ** 2 - pi * radii[-1] ** 2
for i, r in enumerate(radii):
    zcylinders.append(openmoc.ZCylinder(
        x=0.0, y=0.0, radius=r, name='zcylinder%i' % (i)))
    left = openmoc.XPlane(x=-half_pitch, name='left')
    right = openmoc.XPlane(x=half_pitch, name='right')
    top = openmoc.YPlane(y=half_pitch, name='top')
    bottom = openmoc.YPlane(y=-half_pitch, name='bottom')
    left.setBoundaryType(openmoc.REFLECTIVE)
    right.setBoundaryType(openmoc.REFLECTIVE)
    top.setBoundaryType(openmoc.REFLECTIVE)
    bottom.setBoundaryType(openmoc.REFLECTIVE)


###############################################################################
#                             Creating Cells and Universes
###############################################################################
openmoc.log.py_printf('NORMAL', 'Creating cells...')

root_universe = openmoc.Universe(name='root universe')
cells = []
for i in range(n_radii):
    cell = openmoc.Cell(id=i+1)
    cells.append(cell)
    cell.setNumRings(1)
    cell.setNumSectors(1)
    cell.setFill(materials[i])
    cell.addSurface(halfspace=-1, surface=zcylinders[i])
    if i > 0:
        cell.addSurface(halfspace=1, surface=zcylinders[i-1])
    root_universe.addCell(cell)
cell = openmoc.Cell(id=n_radii+1)
cells.append(cell)
cell.setNumSectors(8)
cell.setFill(materials[n_radii])
cell.addSurface(halfspace=1, surface=zcylinders[n_radii-1])
cell.addSurface(halfspace=+1, surface=left)
cell.addSurface(halfspace=-1, surface=right)
cell.addSurface(halfspace=+1, surface=bottom)
cell.addSurface(halfspace=-1, surface=top)
root_universe.addCell(cell)

###############################################################################
#                         Creating the Geometry
###############################################################################
openmoc.log.py_printf('NORMAL', 'Creating geometry...')

geometry = openmoc.Geometry()
geometry.setRootUniverse(root_universe)

###############################################################################
#                          Creating the TrackGenerator
###############################################################################
openmoc.log.py_printf('NORMAL', 'Initializing the track generator...')

track_generator = openmoc.TrackGenerator(geometry, opts.num_azim,
                                         opts.azim_spacing)
track_generator.setNumThreads(opts.num_omp_threads)
track_generator.generateTracks()

###############################################################################
#                            Creating Solver
###############################################################################
solver = openmoc.CPUSolver(track_generator)
solver.setNumThreads(opts.num_omp_threads)
solver.setConvergenceThreshold(opts.tolerance)

###############################################################################
#                            Map Fsr to Material Index
###############################################################################
n_fsr = geometry.getNumFSRs()
fsr2mat = np.zeros(n_fsr, dtype=int)
for ifsr in range(n_fsr):
    fsr2mat[ifsr] = geometry.findFSRMaterial(ifsr).getId() - 1

###############################################################################
#                            Set External Source
###############################################################################
lib = LibraryMicro()
lib.load_from_h5(
    os.path.join(os.getenv('HOME'),
                 'Dropbox/work/codes/openmoc/micromgxs/',
                 'jeff-3.2-wims69e.h5'))
source = np.zeros(ng)
for ig in range(ng):
    erg0 = lib.group_boundaries[ig+1]
    erg1 = lib.group_boundaries[ig]
    source[ig] = quad(lambda x: exp(-x / 0.988e6) * sinh(sqrt(2.249e-6 * x)),
                      erg0, erg1)[0]
sum_source = np.sum(source) * pi * R ** 2
source /= sum_source
for ig in range(ng):
    for ireg in range(n_radii):
        solver.setFixedSourceByCell(cells[ireg], group=ig+1, source=source[ig])


def matid_to_imat(matid):
    for imat in range(n_region):
        if materials[imat].getId() == matid:
            return imat


def get_flux():
    n_fsr = geometry.getNumFSRs()
    flux = np.zeros((ng, n_region))
    fsr_fluxes = openmoc.process.get_scalar_fluxes(solver)
    V = np.zeros(n_region)
    for ifsr in range(n_fsr):
        matid = geometry.findFSRMaterial(ifsr).getId()
        imat = matid_to_imat(matid)
        v = solver.getFSRVolume(ifsr)
        V[imat] += v
        flux[:, imat] += fsr_fluxes[ifsr, :] * v
    flux /= V
    return flux

###############################################################################
#                            Initialize SPH correction
###############################################################################
sph = np.ones((ng, n_region))

# Get macro XS
mac_tot = np.zeros((ng, n_region))
mac_sca = np.zeros((ng, ng, n_region))
with h5py.File(fname) as f:
    for imat in range(n_region):
        mac_tot[:, imat] = f['/material/%i/total' % imat].value[...]
        mac_sca[:, :, imat] \
            = f['/material/%i/scatter matrix' % imat].value[...].reshape(ng, ng)

###############################################################################
#                            SPH correction
###############################################################################
# openmoc.log.set_log_level('TITLE')
# sp = openmc.StatePoint(spfile)
# flux_ref = sp.get_tally(scores=['flux']).mean[:, 0, 0].reshape(ng, n_region)
# for imat in range(n_region):
#     flux_ref[:, imat] = flux_ref[:, imat][::-1] / vols[imat]
# n_iter = 0
# mac_sca_tmp = np.ones((ng, ng, n_region))
# mac_tot_tmp = np.ones((ng, n_region))
# while True:
#     n_iter += 1
#     print(n_iter)
#     if n_iter == 20:
#         print('n_iter > 20')
#         break
#     mac_tot_tmp = mac_tot * sph
#     for ig in range(ng):
#         for imat in range(n_region):
#             mac_sca_tmp[ig, :, imat] *= sph[ig, imat]
#     for imat in range(n_region):
#         materials[imat].setSigmaT(mac_tot[:, imat])
#         materials[imat].setSigmaS(mac_sca[:, :, imat].flatten())
#     solver.computeSource(opts.max_iters, openmoc.SCALAR_FLUX)
#     flux_tmp = get_flux()
#     sph_tmp = flux_ref / flux_tmp
#     sph_err = abs(sph - sph_tmp)
#     sph = sph_tmp
#     if np.max(sph_err) < 1e-4:
#         break
# openmoc.log.set_log_level('NORMAL')

###############################################################################
#                            Solve the Problem
###############################################################################
solver.computeSource(opts.max_iters, openmoc.SCALAR_FLUX)
solver.printTimerReport()

###############################################################################
#                            Get Scalar Flux
###############################################################################
with h5py.File('%s-fs.h5' % name, 'w') as f:
    f['flux'] = get_flux()
    f['sph'] = sph
