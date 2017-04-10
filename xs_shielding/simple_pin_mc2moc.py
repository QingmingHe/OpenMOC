#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import h5py
import openmc
from math import pi

R = 0.41
V = np.zeros(11) + pi * R ** 2 / 10.0
V[-1] = 1.26 ** 2 - pi * R ** 2
sp = openmc.StatePoint('one_nuc_multi_temp/chord_ave/statepoint.2000.h5')
ng = 69
n_reg = 11
flux = sp.get_tally(scores=['flux']).mean[:, 0, 0].reshape(ng, n_reg)
xs_tot = sp.get_tally(scores=['total']).mean[:, 0, 0].reshape(ng, n_reg)
xs_abs = np.zeros((ng, n_reg))
xs_abs[:, :n_reg-1] \
    = sp.get_tally(scores=['absorption']).mean[:, 0, 0].reshape(ng, n_reg-1)
xs_tot /= flux
xs_abs /= flux * 2.21546e-2
xs_sca_0 = sp.get_tally(
    scores=['nu-scatter-0']).mean[:, 0, 0].reshape(ng, ng, n_reg)
for ig in range(ng):
    for ireg in range(n_reg):
        xs_sca_0[ig, :, ireg] /= flux[ig, ireg]
for ireg in range(n_reg):
    flux[:, ireg] = flux[:, ireg][::-1] / V[ireg]
    xs_tot[:, ireg] = xs_tot[:, ireg][::-1]
    xs_abs[:, ireg] = xs_abs[:, ireg][::-1]
    xs_sca_0[:, :, ireg] = xs_sca_0[:, :, ireg].flatten()[::-1].reshape(ng, ng)

f = h5py.File('simple-pin-mc-macro.h5', 'w')
f.attrs['# groups'] = ng
for ireg in range(n_reg):
    f['/material/%i/chi' % ireg] = np.zeros(ng)
    f['/material/%i/total' % ireg] = xs_tot[:, ireg]
    f['/material/%i/fission' % ireg] = np.zeros(ng)
    f['/material/%i/nu-fission' % ireg] = np.zeros(ng)
    f['/material/%i/scatter matrix' % ireg] \
        = xs_sca_0[:, :, ireg].flatten()
f.close()

with h5py.File('ce_calc_ave.h5', 'w') as f:
    f['flux'] = flux
    f['xs_abs'] = xs_abs
    f['r_abs'] = xs_abs * flux
