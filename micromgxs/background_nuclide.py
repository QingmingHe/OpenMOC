#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from math import log10
import xml.etree.ElementTree as et
import xml.dom.minidom as minidom
from potentials import average_potentials


def _copy_attrs(from_grp, to_grp):
    for key in from_grp.attrs:
        to_grp.attrs[key] = from_grp.attrs[key]


def build_background_library(cross_sections, bnuc_file,
                             bnuc_cross_sections=None):
    direct = os.path.dirname(os.path.abspath(cross_sections))
    tree = et.parse(cross_sections)
    root = tree.getroot()
    if bnuc_cross_sections is None:
        bnuc_root = root
    else:
        bnuc_root = et.Element('cross_sections')

    # Process each nuclide
    for library in root:
        if library.attrib['type'] == 'neutron':
            from_h5 = os.path.join(direct, library.attrib['path'])
            from_nuc = library.attrib['materials']
            to_nuc = from_nuc + 'b'
            potential = average_potentials(from_nuc)
            if potential is not None:
                if potential > 0.0:
                    print 'processing {0} ...'.format(to_nuc)
                    build_background_nuclide(from_h5, bnuc_file, from_nuc,
                                             to_nuc, potential)
                    elem = et.SubElement(bnuc_root, 'library')
                    elem.attrib = {'materials': to_nuc, 'path': bnuc_file,
                                   'type': 'neutron'}

    # Export cross sections for background nuclides
    cross_sections_str = et.tostring(bnuc_root)
    cross_sections_str = minidom.parseString(cross_sections_str)
    cross_sections_str = cross_sections_str.toprettyxml()
    if bnuc_cross_sections is None:
        new_cross_sections = cross_sections
    else:
        new_cross_sections = bnuc_cross_sections
    with open(new_cross_sections, 'w') as f:
        f.write(cross_sections_str)


def build_background_nuclide(from_h5, to_h5, from_nuc, to_nuc, potential,
                             n_erg=100, erg_stt=1e-5, erg_end=20e6):
    to_f = h5py.File(to_h5)
    from_f = h5py.File(from_h5)

    # Create nuclide group
    if to_nuc in to_f:
        del to_f[to_nuc]
    to_f.create_group(to_nuc)

    # Copy file attributes
    _copy_attrs(from_f, to_f)

    # Copy nuclide attributes
    _copy_attrs(from_f[from_nuc], to_f[to_nuc])

    # New energy with original attributes
    energy = np.logspace(log10(erg_stt), log10(erg_end), n_erg, True)
    to_f[to_nuc].create_group('energy')
    for temp in from_f[from_nuc]['energy']:
        to_f[to_nuc]['energy'][temp] = energy
    _copy_attrs(from_f[from_nuc]['energy'], to_f[to_nuc]['energy'])

    # copy kTs
    from_f[from_nuc].copy('kTs', to_f[to_nuc])

    # Copy reactions attributes
    to_f[to_nuc].create_group('reactions')
    _copy_attrs(from_f[from_nuc]['reactions'], to_f[to_nuc]['reactions'])

    # Copy attributes of reaction_002
    to_f[to_nuc]['reactions'].create_group('reaction_002')
    _copy_attrs(from_f[from_nuc]['reactions']['reaction_002'],
                to_f[to_nuc]['reactions']['reaction_002'])

    for grp in from_f[from_nuc]['reactions']['reaction_002']:
        if grp.endswith('K'):
            to_f[to_nuc]['reactions']['reaction_002'].create_group(grp)
            # Copy grp attributes
            _copy_attrs(
                from_f[from_nuc]['reactions']['reaction_002'][grp],
                to_f[to_nuc]['reactions']['reaction_002'][grp])
            # Change xs of elastic reaction to potential
            to_f[to_nuc]['reactions']['reaction_002'][grp]['xs'] \
                = np.zeros(n_erg) + potential
            # Copy attributes of xs
            _copy_attrs(
                from_f[from_nuc]['reactions']['reaction_002'][grp]['xs'],
                to_f[to_nuc]['reactions']['reaction_002'][grp]['xs'])
            # Change threshold index to one
            to_f[to_nuc]['reactions']['reaction_002'][grp]['xs'].\
                attrs['threshold_idx'] = 1
        else:
            # Copy other groups
            from_f[from_nuc]['reactions']['reaction_002'].copy(
                grp, to_f[to_nuc]['reactions']['reaction_002'])

    from_f.close()
    to_f.close()

if __name__ == '__main__':
    cross_sections = os.getenv('OPENMC_CROSS_SECTIONS')
    build_background_library(cross_sections, 'background.h5', 'background.xml')
