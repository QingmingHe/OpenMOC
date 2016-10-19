#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt

_NO_RESONANCE = 0.998
_N_BAND_BEGIN = 2
_N_BAND_END = 7


def unify_sub_wgt(pts):
    # Calculate unified accumulated weights
    wgts_acc = []
    unif_acc = []
    for i, pt in enumerate(pts):
        wgts_acc.append([])
        acc = wgts_acc[i]
        if pt['n_band'] > 1:
            acc.append(pt['sub_wgt'][0])
            for ib in range(1, pt['n_band']):
                acc.append(acc[ib-1] + pt['sub_wgt'][ib])
            unif_acc.extend(acc)
    unif_acc = sorted(list(set(unif_acc)))
    n_band = len(unif_acc)

    # Calculate unified subgroup weights
    unif_wgt = np.zeros(n_band)
    unif_wgt[0] = unif_acc[0]
    for ib in range(1, n_band):
        unif_wgt[ib] = unif_acc[ib] - unif_acc[ib-1]

    # Unify subgroup weights
    for i, pt in enumerate(pts):
        if pt['n_band'] > 1:
            sub_tot = np.zeros(n_band)
            sub_abs = np.zeros(n_band)
            sub_sca = np.zeros(n_band)
            sub_nfi = np.zeros(n_band)
            ib0 = 0
            ib1 = 0
            while True:
                sub_tot[ib1] = pt['sub_tot'][ib0]
                sub_abs[ib1] = pt['sub_abs'][ib0]
                sub_sca[ib1] = pt['sub_sca'][ib0]
                if 'sub_nfi' in pt:
                    sub_nfi[ib1] = pt['sub_nfi'][ib0]
                if wgts_acc[i][ib0] < unif_acc[ib1]:
                    ib0 += 1
                elif wgts_acc[i][ib0] == unif_acc[ib1]:
                    ib0 += 1
                    ib1 += 1
                else:
                    ib1 += 1
                if ib0 == pt['n_band'] and ib1 == n_band:
                    break
            pt['n_band'] = n_band
            pt['sub_tot'] = sub_tot
            pt['sub_abs'] = sub_abs
            pt['sub_sca'] = sub_sca
            pt['sub_wgt'] = unif_wgt
            if 'sub_nfi' in pt:
                pt['sub_nfi'] = sub_nfi


class ProbTable(object):

    def __init__(self):
        # Parameters given by user
        self._gc_factor = None
        self._potential = None
        self._dilutions = None
        self._xs_abs = None
        self._xs_sca = None
        self._xs_nfi = None

        # Parameters to be calculated
        self._xs_tot = None
        self._has_prob_table = False
        self._n_band = 0
        self._sub_tot = None
        self._sub_abs = None
        self._sub_sca = None
        self._sub_nfi = None
        self._sub_wgt = None

    def fit(self):
        # Calculate total xs from absorb and scatter
        self._xs_tot = self._xs_sca + self._xs_abs

        # Initialize subgroup parameters
        sub_int = {}
        sub_wgt = {}
        sub_tot = {}
        sub_sca = {}
        sub_abs = {}
        sub_nfi = {}

        # Check whether has resonance
        if self._xs_abs[0] / self._xs_abs[-1] >= _NO_RESONANCE:
            return

        # Fit subgroup parameters from 2 bands to 6 bands
        min_rms = np.Infinity
        min_rms_n_band = 0
        for n_band in range(_N_BAND_BEGIN, _N_BAND_END):
            # Get background cross sections for fit
            idx0, idx1, idx2, backgrounds0, backgrounds1, backgrounds2 \
                = _get_backgrounds(n_band, self._gc_factor, self._potential,
                                   self._dilutions)

            # Fit based on IR model
            sub_int[n_band], sub_wgt[n_band], sub_tot[n_band], \
                sub_sca[n_band], sub_abs[n_band], sub_nfi[n_band] \
                = self._fit_ir(n_band, idx0, idx1, backgrounds0, backgrounds1)

            # Check stability of subgroup parameters
            stable = True
            for ib in range(n_band):
                if sub_tot[n_band][ib] <= 0.0 or sub_sca[n_band][ib] <= 0.0 \
                   or sub_tot[n_band][ib] < sub_sca[n_band][ib] \
                   or sub_wgt[n_band][ib] <= 0.0 or sub_wgt[n_band][ib] >= 1.0:
                    stable = False
                    break

            # Calculate fit error
            if stable:
                n_back2 = len(backgrounds2)
                xs_abs2 = np.zeros(n_back2)
                for i in range(n_back2):
                    xs_abs2[i] = self._xs_abs[idx2[i]]
                rms = _calc_fit_error(xs_abs2, backgrounds2, sub_int[n_band],
                                      sub_abs[n_band], sub_wgt[n_band])
                if rms < min_rms:
                    min_rms_n_band = n_band

        # Get subgroup parameters with min rms
        if min_rms_n_band != 0:
            self._n_band = min_rms_n_band
            self._has_prob_table = True
            self._sub_tot = sub_tot[min_rms_n_band]
            self._sub_sca = sub_sca[min_rms_n_band]
            self._sub_abs = sub_abs[min_rms_n_band]
            self._sub_wgt = sub_wgt[min_rms_n_band]
            if self._xs_nfi is not None:
                self._sub_nfi = sub_nfi[min_rms_n_band]

    def _fit_ir(self, n_band, idx0, idx1, backgrounds0, backgrounds1):
        n_back0 = len(backgrounds0)
        n_back1 = len(backgrounds1)

        # Calculate intermediate xs
        xs_int = self._xs_abs + self._gc_factor  \
            * (self._xs_sca - self._potential)

        # Get xs for fit
        xs_int0 = np.zeros(n_back0)
        xs_tot1 = np.zeros(n_back1)
        xs_sca1 = np.zeros(n_back1)
        xs_abs1 = np.zeros(n_back1)
        if self._xs_nfi is not None:
            xs_nfi1 = np.zeros(n_back1)
        for i in range(n_back0):
            xs_int0[i] = xs_int[idx0[i]]
        for i in range(n_back1):
            xs_tot1[i] = self._xs_tot[idx1[i]]
            xs_sca1[i] = self._xs_sca[idx1[i]]
            xs_abs1[i] = self._xs_abs[idx1[i]]
            if self._xs_nfi is not None:
                xs_nfi1[i] = self._xs_nfi[idx1[i]]

        # Get xs at infinite background
        xs_int_inf = xs_int[-1]
        xs_tot_inf = self._xs_tot[-1]
        xs_sca_inf = self._xs_sca[-1]
        xs_abs_inf = self._xs_abs[-1]
        if self._xs_nfi is not None:
            xs_nfi_inf = self._xs_nfi[-1]

        # Calculate c and d
        c, d = _calc_c_d_coeff(xs_int0, xs_int_inf, backgrounds0, n_band)

        # Calculate subgroup intermediate xs and subgroup weight
        sub_int, sub_wgt = _calc_sub_int_wgt(c, d)

        # Calculate subgroup total xs
        sub_tot = _calc_sub_x(xs_tot1, xs_tot_inf, sub_int, d, backgrounds1)

        # Calculate subgroup scatter xs
        sub_sca = _calc_sub_x(xs_sca1, xs_sca_inf, sub_int, d, backgrounds1)

        # Calculate subgroup absorption xs
        sub_abs = _calc_sub_x(xs_abs1, xs_abs_inf, sub_int, d, backgrounds1)

        # Calculate subgroup nu fission xs
        sub_nfi = None
        if self._xs_nfi is not None:
            sub_nfi = _calc_sub_x(xs_nfi1, xs_nfi_inf, sub_int, d,
                                  backgrounds1)

        return sub_int, sub_wgt, sub_tot, sub_sca, sub_abs, sub_nfi

    @property
    def gc_factor(self):
        return self._gc_factor

    @gc_factor.setter
    def gc_factor(self, gc_factor):
        self._gc_factor = gc_factor

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = potential

    @property
    def dilutions(self):
        return self._dilutions

    @dilutions.setter
    def dilutions(self, dilutions):
        self._dilutions = dilutions

    @property
    def xs_abs(self):
        return self._xs_abs

    @xs_abs.setter
    def xs_abs(self, xs_abs):
        self._xs_abs = xs_abs

    @property
    def xs_sca(self):
        return self._xs_sca

    @xs_sca.setter
    def xs_sca(self, xs_sca):
        self._xs_sca = xs_sca

    @property
    def xs_nfi(self):
        return self._xs_nfi

    @xs_nfi.setter
    def xs_nfi(self, xs_nfi):
        self._xs_nfi = xs_nfi

    @property
    def xs_tot(self):
        return self._xs_tot

    @property
    def n_band(self):
        return self._n_band

    @property
    def sub_tot(self):
        return self._sub_tot

    @property
    def sub_abs(self):
        return self._sub_abs

    @property
    def sub_sca(self):
        return self._sub_sca

    @property
    def sub_nfi(self):
        return self._sub_nfi

    @property
    def sub_wgt(self):
        return self._sub_wgt

    @property
    def has_prob_table(self):
        return self._has_prob_table


def _find_nearest_array(array0, array1):
    n = len(array1)
    array2 = np.zeros(n)
    idx = np.zeros(n)
    for i, val in enumerate(array1):
        idx[i] = np.abs(array0 - val).argmin()
        array2[i] = array0[idx[i]]
    return idx, array2


def _get_backgrounds(n_band, gc_factor, potential, dilutions):
    # Calculate lambda * potential
    n_back = len(dilutions)
    lp = gc_factor * potential

    # Get backgrounds to calculate intermediate subgroup xs and wgt
    if n_band == 2:
        backgrounds0 = [10.0, 52.0]
    elif n_band == 3:
        backgrounds0 = [1e1, 28.0, 52.0, 200.0]
    elif n_band == 4:
        backgrounds0 = [1e1, 28.0, 52.0, 2e2, 1200.0, 1e4]
    elif n_band == 5:
        backgrounds0 = [1e1, 28.0, 52.0, 1e2, 2e2, 6e2, 1e3, 1e4]
    elif n_band == 6:
        backgrounds0 = [1e1, 28.0, 40.0, 52.0, 1e2, 2e2, 6e2, 1e3, 1200.0, 1e4]
    else:
        raise Exception('n_band should be smaller than 6')
    idx0, backgrounds0 = _find_nearest_array(dilutions, backgrounds0)
    backgrounds0 += lp

    # Get backgrounds to fit partial subgroup xs
    backgrounds1 = [5.0, 1e1, 15.0, 20.0, 25.0, 28.0, 30.0, 35.0, 40.0, 45.0,
                    50.0, 52.0, 60.0, 70.0, 80.0, 1e2, 2e2, 4e2, 6e2, 1e3]
    idx1, backgrounds1 = _find_nearest_array(dilutions, backgrounds1)
    backgrounds1 += lp

    # Get backgrounds to calculate fitting error
    idx2 = range(n_back-1)
    backgrounds2 = np.zeros(n_back-1)
    backgrounds2[:] = dilutions[:-1]
    backgrounds2 += lp

    return idx0, idx1, idx2, backgrounds0, backgrounds1, backgrounds2


def _calc_c_d_coeff(xs_int0, xs_int_inf, backgrounds0, n_band):
    L = 2 * n_band - 2
    A = np.zeros((L, L))
    b = np.zeros(L)

    for j in range(L):
        for i in range(n_band - 1):
            A[j, i] = backgrounds0[j] ** (n_band - i - 2)
        for i in range(n_band - 1, L):
            A[j, i] = -xs_int0[j] * backgrounds0[j] ** (L - i - 1)
        b[j] = (xs_int0[j] - xs_int_inf) * backgrounds0[j] ** (n_band - 1)

    x = np.linalg.solve(A, b)

    c = np.zeros(n_band)
    d = np.zeros(n_band)
    c[0] = xs_int_inf
    c[1:n_band] = x[:n_band-1]
    d[0] = 1.0
    d[1:n_band] = x[n_band-1:L]

    return c, d


def _calc_sub_int_wgt(c, d):
    n_band = len(c)

    coeff = np.zeros(n_band+1)
    coeff[0] = d[0]
    coeff[-1] = c[-1]
    for i in range(1, n_band):
        coeff[i] = d[i] + c[i-1]

    sub_int = -np.roots(coeff).real
    sub_wgt = np.zeros(n_band)

    for i in range(n_band):
        numerator = 0.0
        denominator = 1.0
        for j in range(n_band):
            numerator += d[j] * (-sub_int[i]) ** (n_band - j - 1)
            if j != i:
                denominator *= (sub_int[j] - sub_int[i])
        if denominator == 0.0:
            sub_wgt[i] = -np.Infinity
        else:
            sub_wgt[i] = numerator / denominator

    return sub_int, sub_wgt


def _calc_sub_x(xs_x1, xs_x_inf, sub_int, d, backgrounds1):
    n_band = len(d)
    n_back = len(backgrounds1)

    A = np.zeros((n_band-1, n_band-1))
    for i in range(n_band-1):
        for j in range(n_band-1):
            for k in range(n_back):
                A[i, j] += backgrounds1[k] ** (i + j)

    b = np.zeros(n_band-1)
    for i in range(n_band-1):
        for k in range(n_back):
            G = 0.0
            for j in range(n_band):
                G += d[j] * backgrounds1[k] ** (n_band - j - 1)
            G = G * xs_x1[k] - xs_x_inf * backgrounds1[k] ** (n_band-1)
            b[i] += G * backgrounds1[k] ** i

    e_bar = np.linalg.solve(A, b)

    e = np.zeros(n_band)
    e[0] = xs_x_inf
    for i in range(1, n_band):
        e[i] = e_bar[n_band-i-1]

    sub_x = np.zeros(n_band)
    for i in range(n_band):
        numerator = 0.0
        denominator = 0.0
        for j in range(n_band):
            sub_int_x = (-sub_int[i]) ** (n_band-j+1)
            numerator += sub_int_x * e[j]
            denominator += sub_int_x * d[j]
        if denominator == 0.0:
            sub_x[i] = -np.Infinity
        else:
            sub_x[i] = numerator / denominator

    return sub_x


def _calc_fit_error(xs_x2, backgrounds2, sub_int, sub_x, sub_wgt):
    n_back = len(backgrounds2)
    rms = 0.0
    err = np.zeros(n_back)
    for i in range(n_back):
        sub_flux = sub_wgt * backgrounds2[i] / (sub_int + backgrounds2[i])
        if xs_x2[i] == 0.0:
            err[i] = 0.0
        else:
            err[i] = (np.sum(sub_x * sub_flux) / np.sum(sub_flux) - xs_x2[i]) \
                     / xs_x2[i]
        rms += err[i] ** 2
    rms = sqrt(rms / n_back)

    return rms


def test_fit():
    p = ProbTable()
    p.gc_factor = 2.5027629999999999E-002
    p.potential = 11.293448000000000
    p.dilutions = np.array([
        5.0000000000000000, 10.000000000000000, 15.000000000000000,
        20.000000000000000, 25.000000000000000, 28.000000000000000,
        30.000000000000000, 35.000000000000000, 40.000000000000000,
        45.000000000000000, 50.000000000000000, 52.000000000000000,
        55.000000000000000, 60.000000000000000, 70.000000000000000,
        80.000000000000000, 100.00000000000000, 120.00000000000000,
        140.00000000000000, 160.00000000000000, 200.00000000000000,
        260.00000000000000, 400.00000000000000, 600.00000000000000,
        800.00000000000000, 1000.0000000000000, 1200.0000000000000,
        2000.0000000000000, 3500.0000000000000, 5000.0000000000000,
        7000.0000000000000, 10000.000000000000, 100000.00000000000,
        10000000000.000000
    ])
    p.xs_abs = np.array([
        3.1681599999999999, 4.0505599999999999, 4.7431900000000002,
        5.3352199999999996, 5.8624999999999998, 6.1559200000000001,
        6.3436800000000000, 6.7899099999999999, 7.2084700000000002,
        7.6044299999999998, 7.9814900000000000, 8.1276700000000002,
        8.3424300000000002, 8.6894100000000005, 9.3480500000000006,
        9.9677000000000007, 11.116600000000000, 12.173299999999999,
        13.159900000000000, 14.090600000000000, 15.821300000000001,
        18.174399999999999, 22.927199999999999, 28.617799999999999,
        33.490000000000002, 37.789700000000003, 41.652000000000001,
        54.110199999999999, 70.025300000000001, 80.783900000000003,
        90.821600000000004, 100.83499999999999, 135.93100000000001,
        142.01100000000000
    ])
    p.xs_sca = np.array([
        9.3101400000000005, 9.2882400000000001, 9.2946100000000005,
        9.3085799999999992, 9.3254000000000001, 9.3360800000000008,
        9.3433200000000003, 9.3615899999999996, 9.3799299999999999,
        9.3981700000000004, 9.4161099999999998, 9.4232300000000002,
        9.4338700000000006, 9.4513900000000000, 9.4854500000000002,
        9.5184999999999995, 9.5816999999999997, 9.6415000000000006,
        9.6982999999999997, 9.7528000000000006, 9.8554999999999993,
        9.9974000000000007, 10.289199999999999, 10.643300000000000,
        10.948900000000000, 11.219700000000000, 11.463600000000000,
        12.253200000000000, 13.265499999999999, 13.951400000000000,
        14.592400000000000, 15.231999999999999, 17.478000000000002,
        17.867999999999999
    ])

    p.fit()
    print p.n_band
    print p.sub_tot
    print p.sub_wgt


def test_unify():
    pt0 = {'n_band': 3, 'sub_wgt': np.array([0.1, 0.4, 0.5]),
           'sub_tot': np.array([1e4, 1e3, 1e2]),
           'sub_abs': np.array([1e4, 1e3, 1e2]),
           'sub_sca': np.array([1e4, 1e3, 1e2])}
    pt1 = {'n_band': 4, 'sub_wgt': np.array([0.1, 0.2, 0.3, 0.4]),
           'sub_tot': np.array([1e4, 1e3, 1e2, 1e1]),
           'sub_abs': np.array([1e4, 1e3, 1e2, 1e1]),
           'sub_sca': np.array([1e4, 1e3, 1e2, 1e1])}
    pt2 = {'n_band': 5, 'sub_wgt': np.array([0.05, 0.15, 0.2, 0.25, 0.35]),
           'sub_tot': np.array([1e4, 1e3, 1e2, 1e1, 1.0]),
           'sub_abs': np.array([1e4, 1e3, 1e2, 1e1, 1.0]),
           'sub_sca': np.array([1e4, 1e3, 1e2, 1e1, 1.0])}
    pts = [pt0, pt1, pt2]
    unify_sub_wgt(pts)
    from pprint import pprint
    pprint(pts)

if __name__ == '__main__':
    test_unify()
