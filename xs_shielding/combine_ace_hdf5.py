#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py


def _get_nuclide(fname):
    with h5py.File(fname) as f:
        return f.keys()[0]


def combine(files, to_file):
    nuclide = _get_nuclide(files[0])
    with h5py.File(to_file, 'w') as f1:
        f1.create_group(nuclide)
        f1[nuclide].create_group('energy')
        f1[nuclide].create_group('kTs')
        f1[nuclide].create_group('reactions')
        f1[nuclide].create_group('urr')
        for ifile, afile in enumerate(files):
            with h5py.File(afile) as f0:
                if ifile == 0:
                    for key in f0[nuclide].attrs:
                        f1[nuclide].attrs[key] = f0[nuclide].attrs[key]
                grp = '/%s/energy' % nuclide
                for kT in f0[grp]:
                    if kT not in f1[grp]:
                        f0.copy('%s/%s' % (grp, kT), f1[grp])
                grp = '/%s/kTs' % nuclide
                for kT in f0[grp]:
                    if kT not in f1[grp]:
                        f0.copy('%s/%s' % (grp, kT), f1[grp])
                for reaction in f0[nuclide]['reactions']:
                    if ifile == 0:
                        f0.copy('/%s/reactions/%s' % (nuclide, reaction),
                                f1[nuclide]['reactions'])
                    else:
                        for kT in f0[nuclide]['reactions'][reaction]:
                            if kT not in f1[nuclide]['reactions'][reaction]:
                                f0.copy('/%s/reactions/%s/%s' %
                                        (nuclide, reaction, kT),
                                        f1[nuclide]['reactions'][reaction])
                if ifile == 0:
                    f0.copy('/%s/total_nu' % nuclide, f1[nuclide])
                grp = '/%s/urr' % nuclide
                for kT in f0[grp]:
                    if kT not in f1[grp]:
                        f0.copy('%s/%s' % (grp, kT), f1[grp])

if __name__ == '__main__':
    import os
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    jeff_dir = os.path.dirname(cross_sections)
    u238_files = [os.path.join(jeff_dir, fname)
                  for fname in ['U238.h5', 'U238new.h5']]
    u238all = os.path.join(jeff_dir, 'U238all.h5')
    combine(u238_files, u238all)
