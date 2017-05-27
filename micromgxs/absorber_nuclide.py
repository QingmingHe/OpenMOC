#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py

reactions = '/H1a/reactions'
fname = "/home/qmhe/library/jeff-3.2-1.0/jeff-3.2-hdf5/H1.h5"
with h5py.File(fname) as f:
    for reaction in f[reactions]:
        for temp in f[reactions][reaction]:
            if temp.endswith('K'):
                if reaction == 'reaction_102':
                    f[reactions][reaction][temp]['xs'][...] = 20.478001
                else:
                    f[reactions][reaction][temp]['xs'][...] = 0.0
