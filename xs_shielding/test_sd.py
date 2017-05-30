#!/usr/bin/env python
# -*- coding: utf-8 -*-
import openmoc
import os
import numpy as np
from openmoc.library_ce import LibraryCe
from math import log

# Define the calculation parameters
dilution = 50.0
dilutions = [5.0, 1e1, 15.0, 20.0, 25.0, 28.0, 30.0, 35.0, 40.0,
             45.0, 50.0, 52.0, 60.0, 70.0, 80.0, 1e2, 2e2, 4e2,
             6e2, 1e3, 1.2e3, 1e4, 1e10]
n_dilution = len(dilutions)
dmu = 1e-5
erg_bnd = np.array([9.1180E+03, 5.5300E+03, 3.5191E+03,
                    2.2394E+03, 1.4251E+03, 9.0690E+02, 3.6726E+02, 1.4873E+02,
                    7.5501E+01, 4.8052E+01, 2.7700E+01, 1.5968E+01, 9.8770E+00,
                    4.0000E+00])
# erg_bnd = np.array([
#             2.47800E+04, 1.50300E+04, 9.11800E+03,
#             5.53000E+03, 3.51910E+03, 2.23945E+03, 1.42510E+03,
#             9.06899E+02, 750.0, 500.0, 3.67263E+02, 325.0, 225.0, 200.0,
#             1.48729E+02, 110.0, 7.55014E+01, 4.80520E+01, 2.77000E+01,
#             1.59680E+01, 9.87700E+00, 4.00000E+00, 3.30000E+00, 2.60000E+00,
#             2.10000E+00, 1.50000E+00, 1.30000E+00, 1.15000E+00, 1.12300E+00,
# ])
nbg = erg_bnd.shape[0] - 1
n_case = 1
emin = erg_bnd[-1]
emax = erg_bnd[0]

# Get the hyper-fine energy xs for U-238
lib = LibraryCe(os.getenv('JEFF_CROSS_SECTIONS'))
reslib = lib.get_nuclide('Pu241', 293.6, emax, emin, False, dmu=dmu,
                           find_nearest_temp=True)

# Define the slowing down solver
sd = openmoc.SDSolver()
sd.setErgGrpBnd(erg_bnd)
sd.setNumNuclide(2)
sd.setSolErgBnd(emin, emax)

# Define resonnant nuclide
resnuc = sd.getNuclide(0)
resnuc.setName('ResNuc')
resnuc.setHFLibrary(reslib)
resnuc.setNumDens(np.ones(len(dilutions)))

# Define H-1 (background)
h1 = sd.getNuclide(1)
h1.setName('H1')
h1.setAwr(0.9991673)
h1.setPotential(1.0)
h1.setNumDens(dilutions)

# Compute hyper-fine energy flux
sd.computeFlux()

# Compute MG XS
sd.computeMgXs()

idlt = n_dilution - 1
for ig in range(nbg):
    ref_flux = sd.getMgFlux(ig, idlt) * dmu / log(erg_bnd[ig] / erg_bnd[ig+1])
    xs_tot = resnuc.getMgTotal(ig, idlt)
    xs_sca = resnuc.getMgScatter(ig, idlt)
    xs_abs = xs_tot - xs_sca
    print('%2i %f' % (ig, xs_abs))
