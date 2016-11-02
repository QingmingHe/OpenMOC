#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
import numpy as np
from prob_table import ProbTable
from bisect import bisect
from math import sqrt


class NuclideMicro(object):

    def __init__(self):
        # Header data
        self._nuclide = None
        self._A = None
        self._Z = None
        self._awr = None
        self._ng = None
        self._first_res = None
        self._last_res = None

        # Full xs
        self._full_xs_temps = None
        self._legendre_order = None
        self._fissionable = False
        self._xs_tot = None
        self._xs_sca = None
        self._xs_fis = None
        self._xs_nfi = None
        self._xs_abs = None
        self._xs_sca_tot = None
        self._chi = None

        # Resonance table
        self._has_res = False
        self._has_resfis = False
        self._average_lambda = None
        self._average_potential = None
        self._gc_factor = None
        self._potential = None
        self._res_temps = None
        self._res_abs = None
        self._res_sca = None
        self._res_nfi = None
        self._dilutions = None

    def temp_interp_res_tbl(self, temp, ig):
        jg = ig - self._first_res
        if self._has_res:
            # Interpolate resonance xs table
            temps = self._res_temps
            n_temp = len(temps)
            itemp1 = bisect(temps, temp)
            if itemp1 == 0:
                itemp1 = 1
            elif itemp1 == n_temp:
                itemp1 = n_temp - 1
            itemp0 = itemp1 - 1
            r = (sqrt(temps[itemp1]) - sqrt(temp)) \
                / (sqrt(temps[itemp1]) - sqrt(temps[itemp0]))
            res_sca = r * self._res_sca[itemp0][jg, :] \
                + (1.0 - r) * self._res_sca[itemp1][jg, :]
            res_abs = r * self._res_abs[itemp0][jg, :] \
                + (1.0 - r) * self._res_abs[itemp1][jg, :]
            res_tot = res_sca + res_abs
            if self._has_resfis:
                res_nfi = r * self._res_nfi[itemp0][jg, :] \
                    + (1.0 - r) * self._res_nfi[itemp1][jg, :]
            else:
                res_nfi = None
            # Interpolate potential xs
            temps = self._full_xs_temps
            n_temp = len(temps)
            itemp1 = bisect(temps, temp)
            if itemp1 == 0:
                itemp1 = 1
            elif itemp1 == n_temp:
                itemp1 = n_temp - 1
            itemp0 = itemp1 - 1
            r = (sqrt(temps[itemp1]) - sqrt(temp)) \
                / (sqrt(temps[itemp1]) - sqrt(temps[itemp0]))
            potential = r * self._potential[itemp0][jg] \
                + (1.0 - r) * self._potential[itemp1][jg]
            return {'res_tot': res_tot, 'res_abs': res_abs, 'res_sca': res_sca,
                    'res_nfi': res_nfi, 'dilutions': self._dilutions,
                    'lambda': self._gc_factor[jg], 'potential': potential}
        else:
            raise Exception('no resonance for %s' % (self._nuclide))

    def load_from_h5(self, nuclide, fname=None, fh=None):
        self._nuclide = nuclide

        if fname is not None:
            f = h5py.File(fname)
        elif fh is not None:
            f = fh
        else:
            raise Exception('fname or fh should be given')

        # Load header data
        nuc_group = '/' + self._nuclide
        self._A = f[nuc_group].attrs['A']
        self._Z = f[nuc_group].attrs['Z']
        self._awr = f[nuc_group].attrs['awr']
        self._ng = f[nuc_group].attrs['ng']
        self._first_res = f[nuc_group].attrs['first_res']
        self._last_res = f[nuc_group].attrs['last_res']

        # Load full xs data
        fxs_group = nuc_group + '/full_xs'
        if f[fxs_group].attrs['fissionable'] == 1:
            self._fissionable = True
        else:
            self._fissionable = False
        self._legendre_order = f[fxs_group].attrs['legendre_order']
        self._full_xs_temps = f[fxs_group]['temperatures'].value
        n_temp = len(self._full_xs_temps)
        self._xs_tot = {}
        self._xs_sca = {}
        self._xs_abs = {}
        self._xs_sca_tot = {}
        if self._fissionable:
            self._xs_fis = {}
            self._xs_nfi = {}
            self._chi = {}
        for itemp in range(n_temp):
            temp_group = fxs_group + '/temp' + str(itemp)
            self._xs_tot[itemp] = f[temp_group]['total'].value
            self._xs_sca[itemp] = {}
            self._xs_sca_tot[itemp] = np.zeros(self._ng)
            for il in range(self._legendre_order):
                self._xs_sca[itemp][il] = {}
                for ig_to in range(self._ng):
                    scatt_dset = temp_group + '/scatter/lo%s/to%s' %\
                                  (il, ig_to)
                    ig0 = f[scatt_dset].attrs['ig0']
                    ig1 = f[scatt_dset].attrs['ig1']
                    self._xs_sca[itemp][il][ig_to] = {}
                    self._xs_sca[itemp][il][ig_to]['ig0'] = ig0
                    self._xs_sca[itemp][il][ig_to]['ig1'] = ig1
                    self._xs_sca[itemp][il][ig_to]['data'] \
                        = f[scatt_dset].value
                    for ig_from in range(ig0, ig1):
                        xs_sca_data = self._xs_sca[itemp][il][ig_to]['data']
                        self._xs_sca_tot[itemp][ig_from] \
                            += xs_sca_data[ig_from-ig0]
            self._xs_abs[itemp] = self._xs_tot[itemp] - self._xs_sca_tot[itemp]
            if self._fissionable:
                self._xs_fis[itemp] = f[temp_group]['fission'].value
                self._xs_nfi[itemp] = f[temp_group]['nu_fission'].value
                self._chi[itemp] = f[temp_group]['chi'].value

        # Very special case for potential, which is scattering xs at full xs
        # temperatures at typical dilution
        self._potential = {}
        for itemp in range(len(self._full_xs_temps)):
            self._potential[itemp] \
                = self._xs_sca_tot[itemp][self._first_res:self._last_res]

        # Load resonance data
        res_group = nuc_group + '/resonance'
        n_res = self._last_res - self._first_res
        if f[res_group].attrs['has_res'] == 1:
            self._has_res = True
        else:
            self._has_res = False
        if f[res_group].attrs['has_resfis'] == 1:
            self._has_resfis = True
        else:
            self._has_resfis = False
        self._average_lambda = f[res_group]['average_lambda'].value
        self._average_potential = f[res_group]['average_potential'].value
        if 'lambda' in f[res_group]:
            self._gc_factor = f[res_group]['lambda'].value
        else:
            self._gc_factor = np.zeros(n_res) + self._average_lambda
        # Read self-shielded xs
        if self._has_res:
            self._res_abs = {}
            self._res_sca = {}
            if self._has_resfis:
                self._res_nfi = {}
            self._res_temps = f[res_group]['temperatures'].value
            self._dilutions = f[res_group]['dilutions'].value
            n_temp = len(self._res_temps)
            for itemp in range(n_temp):
                temp_group = res_group + '/temp%s' % (itemp)
                self._res_abs[itemp] = f[temp_group]['absorption'].value
                self._res_sca[itemp] = f[temp_group]['scatter'].value
                if self._has_resfis:
                    self._res_nfi[itemp] = f[temp_group]['nu_fission'].value

        if fname is not None:
            f.close()


class LibraryMicro(object):

    def __init__(self):
        # Group structure
        self._ng = None
        self._first_res = None
        self._last_res = None
        self._group_boundaries = None

        # Fission spectrum
        self._average_chi = None

        # Nuclides
        self._nuclides = None

    def load_from_h5(self, fname):
        f = h5py.File(fname)

        # Load group structure
        self._group_boundaries = f['group_structure'].value
        self._ng = len(self._group_boundaries) - 1
        self._first_res = f['group_structure'].attrs['first_res']
        self._last_res = f['group_structure'].attrs['last_res']

        # Load fission spectrum
        self._average_chi = f['fission_spectrum'].value

        # Load nuclides
        self._nuclides = {}
        for nuclide in f:
            is_nuclide = False
            if 'is_nuclide' in f[nuclide].attrs:
                if f[nuclide].attrs['is_nuclide'] == 1:
                    is_nuclide = True
            if is_nuclide:
                self._nuclides[nuclide] = NuclideMicro()
                self._nuclides[nuclide].load_from_h5(nuclide, fh=f)

        f.close()

    def set_gc_factor(self, nuclide, ig, gc_factor):
        jg = ig - self._first_res
        self._nuclides[nuclide]._gc_factor[jg] = gc_factor

    def get_typical_xs(self, nuclide, temp, ig, *args):
        nuc = self._nuclides[nuclide]
        temps = nuc._full_xs_temps
        n_temp = len(temps)
        jg = ig - nuc._first_res
        xs = {}

        # Calculate temperature interpolation ratio
        itemp1 = bisect(nuc._full_xs_temps, temp)
        if itemp1 == 0:
            itemp1 = 1
        elif itemp1 == n_temp:
            itemp1 = n_temp - 1
        itemp0 = itemp1 - 1
        r = (sqrt(temps[itemp1]) - sqrt(temp))\
            / (sqrt(temps[itemp1]) - sqrt(temps[itemp0]))

        # Toal xs
        if 'total' in args:
            xs['total'] = r * nuc._xs_tot[itemp0][ig] \
                + (1.0 - r) * nuc._xs_tot[itemp1][ig]
        # Scatter xs
        if 'scatter' in args:
            xs['scatter'] = r * nuc._xs_sca_tot[itemp0][ig] \
                + (1.0 - r) * nuc._xs_sca_tot[itemp1][ig]
        # Absorb xs
        if 'absorb' in args:
            xs['absorb'] = r * nuc._xs_abs[itemp0][ig] \
                + (1.0 - r) * nuc._xs_abs[itemp1][ig]
        # Nu fission xs
        if 'nufis' in args:
            if nuc._fissionable:
                xs['nufis'] = r * nuc._xs_nfi[itemp0][ig] \
                    + (1.0 - r) * nuc._xs_nfi[itemp1][ig]
            else:
                xs['nufis'] = 0.0
        # Nu
        if 'nu' in args:
            xs['nu'] = 0.0
            if nuc._fissionable:
                nufis = r * nuc._xs_nfi[itemp0][ig] \
                        + (1.0 - r) * nuc._xs_nfi[itemp1][ig]
                fis = r * nuc._xs_fis[itemp0][ig] \
                    + (1.0 - r) * nuc._xs_fis[itemp1][ig]
                if fis > 1e-13:
                    xs['nu'] = nufis / fis
        # XS for resonance groups
        if ig >= nuc._first_res and ig < nuc._last_res:
            # Goldstein-Cohen factor
            if 'lambda' in args:
                xs['lambda'] = nuc._gc_factor[jg]
            if 'potential' in args:
                if nuc._has_res:
                    xs['potential'] = nuc._average_potential
                else:
                    xs['potential'] = r * nuc._potential[itemp0][jg] \
                        + (1.0 - r) * nuc._potential[itemp1][jg]

        return xs

    def get_subp(self, nuclide, temp, ig):
        nuc = self._nuclides[nuclide]

        # Temperature interpolate resonance xs table
        res_xs = nuc.temp_interp_res_tbl(temp, ig)

        # Fit subgroup parameters
        pt = ProbTable()
        pt.gc_factor = res_xs['lambda']
        pt.potential = res_xs['potential']
        pt.dilutions = res_xs['dilutions']
        pt.xs_abs = res_xs['res_abs']
        pt.xs_sca = res_xs['res_sca']
        pt.xs_nfi = res_xs['res_nfi']
        pt.fit()

        return {'lambda': pt.gc_factor, 'potential': pt.potential, 'sub_tot':
                pt.sub_tot, 'sub_abs': pt.sub_abs, 'sub_sca': pt.sub_sca,
                'sub_nfi': pt.sub_nfi, 'sub_wgt': pt.sub_wgt, 'n_band':
                pt.n_band}

    @property
    def ng(self):
        return self._ng

    @property
    def first_res(self):
        return self._first_res

    @property
    def last_res(self):
        return self._last_res

    @property
    def group_boundaries(self):
        return self._group_boundaries

    @property
    def average_chi(self):
        return self._average_chi

    def has_res(self, nuclide):
        return self._nuclides[nuclide]._has_res

    def has_resfis(self, nuclide):
        return self._nuclides[nuclide]._has_resfis

if __name__ == "__main__":
    import os
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmc/openmc/micromgxs/',
                         'jeff-3.2-wims69e.h5')
    lib = LibraryMicro()
    lib.load_from_h5(fname)
    print lib.has_res('U238')
    print lib.last_res - lib.first_res
