#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
import numpy as np

fname = 'simple-pin-975-macro.h5'
n_region = 11
ng = 69
with h5py.File(fname) as f:
    mac_tot = np.zeros((ng, n_region))
    for imat in range(n_region):
        mac_tot[:, imat] = f['/material/%i/total' % imat].value[...]

for ig in range(14, 27):
    print mac_tot[ig, :]
