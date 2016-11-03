# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import h5py
import numpy as np
from math import log, exp
from bisect import bisect
from potentials import average_potential
from openmoc import SDLibrary, interpEnergy, interpEnergyTemperature

_K_BOLTZMANN = 8.6173324e-11
_N_GAMMA = 102
_N_DA = 117
_N_P0 = 600
_N_AC = 849
_N_TA = 155
_N_DT = 182
_N_P3HE = 191
_N_D3HE = 192
_N_3HEA = 193
_N_3P = 197
_N_FISSION = 18
_N_F = 19
_N_NF = 20
_N_2NF = 21
_N_3NF = 38
_ELASTIC = 2


def _find_temp_interp_index(kT, kTs):
    n = len(kTs)
    new_kTs = sorted(kTs)
    i1 = bisect(new_kTs, kT)
    if i1 > n - 1:
        i1 = n - 1
    elif i1 < 1:
        i1 = 1
    i0 = i1 - 1
    i0 = kTs.index(new_kTs[i0])
    i1 = kTs.index(new_kTs[i1])
    return i0, i1


def is_disappearance(mt):
    if mt >= _N_GAMMA and mt <= _N_DA:
        return True
    elif mt >= _N_P0 and mt <= _N_AC:
        return True
    elif mt in (_N_TA, _N_DT, _N_P3HE, _N_D3HE, _N_3P):
        return True
    else:
        return False


def _is_fission(mt):
    if mt in (_N_FISSION, _N_F, _N_NF, _N_2NF, _N_3NF):
        return True
    else:
        return False


def _interp_erg(xs, emax, emin, n, has_res_fis):
    du = log(emax / emin) / n
    energy0 = xs['energy']
    xs_abs0 = xs['xs_abs']
    xs_ela0 = xs['xs_ela']
    if has_res_fis:
        xs_fis0 = xs['xs_fis']

    u = log(emax / emin) - du / 2.0
    e = emax * exp(-u)
    i1 = bisect(energy0, e)
    i0 = i1 - 1
    xs_abs = np.zeros(n)
    xs_ela = np.zeros(n)
    if has_res_fis:
        xs_fis = np.zeros(n)
    for i in range(n):
        r0 = (energy0[i1] - e) / (energy0[i1] - energy0[i0])
        r1 = 1.0 - r0
        xs_ela[i] = r0 * xs_ela0[i0] + r1 * xs_ela0[i1]
        xs_abs[i] = r0 * xs_abs0[i0] + r1 * xs_abs0[i1]
        if has_res_fis:
            xs_fis[i] = r0 * xs_fis0[i0] + r1 * xs_fis0[i1]

        u -= du
        e = emax * exp(-u)
        while True:
            if e < energy0[i1]:
                break
            i1 += 1
            i0 = i1 - 1

    if has_res_fis:
        return {'xs_abs': xs_abs, 'xs_ela': xs_ela, 'xs_fis': xs_fis}
    else:
        return {'xs_abs': xs_abs, 'xs_ela': xs_ela}


def _interp_erg_temp(xs0, xs1, temp0, temp1, emax, emin, n, temp, has_res_fis):
    rt0 = (temp1 - temp) / (temp1 - temp0)
    rt1 = 1.0 - rt0
    du = log(emax / emin) / n
    energy0 = xs0['energy']
    energy1 = xs1['energy']
    xs_abs0 = xs0['xs_abs']
    xs_abs1 = xs1['xs_abs']
    xs_ela0 = xs0['xs_ela']
    xs_ela1 = xs1['xs_ela']
    if has_res_fis:
        xs_fis0 = xs0['xs_fis']
        xs_fis1 = xs1['xs_fis']

    u = log(emax / emin) - du / 2.0
    e = emax * exp(-u)
    i1 = bisect(energy0, e)
    i0 = i1 - 1
    j1 = bisect(energy1, e)
    j0 = j1 - 1
    xs_abs = np.zeros(n)
    xs_ela = np.zeros(n)
    if has_res_fis:
        xs_fis = np.zeros(n)
    for i in range(n):
        r0 = (energy0[i1] - e) / (energy0[i1] - energy0[i0])
        r1 = 1.0 - r0
        xs_abs2 = r0 * xs_abs0[i0] + r1 * xs_abs0[i1]
        xs_ela2 = r0 * xs_ela0[i0] + r1 * xs_ela0[i1]
        if has_res_fis:
            xs_fis2 = r0 * xs_fis0[i0] + r1 * xs_fis0[i1]

        r0 = (energy1[j1] - e) / (energy1[j1] - energy1[j0])
        r1 = 1.0 - r0
        xs_abs3 = r0 * xs_abs1[j0] + r1 * xs_abs1[j1]
        xs_ela3 = r0 * xs_ela1[j0] + r1 * xs_ela1[j1]
        if has_res_fis:
            xs_fis3 = r0 * xs_fis1[j0] + r1 * xs_fis1[j1]

        xs_abs[i] = rt0 * xs_abs2 + rt1 * xs_abs3
        xs_ela[i] = rt0 * xs_ela2 + rt1 * xs_ela3
        if has_res_fis:
            xs_fis[i] = rt0 * xs_fis2 + rt1 * xs_fis3

        u -= du
        e = emax * exp(-u)
        while True:
            if e < energy0[i1]:
                break
            i1 += 1
            i0 = i1 - 1
        while True:
            if e < energy1[j1]:
                break
            j1 += 1
            j0 = j1 - 1

    if has_res_fis:
        return {'xs_abs': xs_abs, 'xs_ela': xs_ela, 'xs_fis': xs_fis}
    else:
        return {'xs_abs': xs_abs, 'xs_ela': xs_ela}


class LibraryCe(object):

    def __init__(self, cross_sections):
        self._cross_sections = cross_sections
        self._nuclides = {}

        direc = os.path.dirname(os.path.abspath(cross_sections))

        tree = ET.parse(cross_sections)
        root = tree.getroot()
        for elem in root:
            if elem.attrib['type'] == 'neutron':
                mat = elem.attrib['materials']
                self._nuclides[mat] = {}
                self._nuclides[mat]['path'] \
                    = os.path.join(direc, elem.attrib['path'])

    def get_nuclide(self, nuclide, temp, emax_eV_MeV, emin_eV_MeV, has_res_fis,
                    dmu=1e-5, energy_unit='eV', find_nearest_temp=True):
        if nuclide not in self._nuclides:
            raise Exception('%s not in cross_sections' % nuclide)

        # Convert energy to MeV
        if energy_unit == 'eV':
            emax = emax_eV_MeV * 1e-6
            emin = emin_eV_MeV * 1e-6
        else:
            emax = emax_eV_MeV
            emin = emin_eV_MeV

        # Compute number of hyper-fine energy groups
        n = int(log(emax / emin) / dmu)

        f = h5py.File(self._nuclides[nuclide]['path'])

        # Get temperatures in file
        kT = temp * _K_BOLTZMANN
        kTs = []
        kT_grps = []
        for grp in f[nuclide]['kTs']:
            kT_grps.append(grp)
            kTs.append(f[nuclide]['kTs'][grp].value)

        if find_nearest_temp:
            # Find nearest temperature point
            ii = (np.abs(np.array(kTs) - kT)).argmin()

            # Only energy point interpolation
            acelib = self.get_xs_by_kT_grp(f, nuclide, kT_grps[ii],
                                           has_res_fis)
            hflib = interpEnergy(acelib, emax, emin, n, has_res_fis)
        else:
            # Find temperature interpolation index
            i0, i1 = _find_temp_interp_index(kT, kTs)

            # Temperature interpolation and energy point interpolation
            acelib0 = self.get_xs_by_kT_grp(f, nuclide, kT_grps[i0],
                                            has_res_fis)
            acelib1 = self.get_xs_by_kT_grp(f, nuclide, kT_grps[i1],
                                            has_res_fis)
            hflib = interpEnergyTemperature(
                acelib0, acelib1, kTs[i0], kTs[i1], kT, emax, emin, n,
                has_res_fis)

        # Get potential and atomic weight ratio
        hflib.setPotential(average_potential(nuclide))
        hflib.setAwr(f[nuclide].attrs['atomic_weight_ratio'])

        f.close()

        return hflib

    def get_xs_by_kT_grp(self, f, nuclide, kT_grp, has_res_fis):
        # Energy points
        energy = f[nuclide]['energy'][kT_grp].value
        n_energy = len(energy)

        # Accumulate cross sections
        xs_tot = np.zeros(n_energy)
        xs_sca = np.zeros(n_energy)
        if has_res_fis:
            xs_fis = np.zeros(n_energy)
        for reaction in f[nuclide]['reactions']:
            mt = int(reaction[9:12])
            xs = f[nuclide]['reactions'][reaction][kT_grp]['xs']
            i = xs.attrs['threshold_idx'] - 1
            n = xs.shape[0]
            if is_disappearance(mt):
                xs_tot[i:i+n] += xs.value
            if mt == _ELASTIC:
                xs_tot[i:i+n] = xs.value
                xs_sca[i:i+n] = xs.value
            if has_res_fis:
                if _is_fission(mt):
                    xs_fis[i:i+n] += xs.value

        acelib = SDLibrary()
        acelib.setEnergy(energy)
        acelib.setXsTotal(xs_tot)
        acelib.setXsScatter(xs_sca)
        if has_res_fis:
            acelib.setXsFission(xs_fis)
        return acelib


if __name__ == '__main__':
    lib = LibraryCe(os.getenv('OPENMC_CROSS_SECTIONS'))
    lib.get_nuclide('U238', 293.6, 9118.0, 4.0, False)
