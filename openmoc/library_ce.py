# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import h5py
import numpy as np
from scipy.io import FortranFile
from math import log, exp
from bisect import bisect
from potentials import average_potential
from openmoc import (SDLibrary, SDLibrary_interpEnergy,
                     SDLibrary_interpEnergyTemperature)

_K_BOLTZMANN = 8.6173324e-5
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
    xs_tot0 = xs['xs_tot']
    xs_sca0 = xs['xs_sca']
    if has_res_fis:
        xs_fis0 = xs['xs_fis']

    u = log(emax / emin) - du / 2.0
    e = emax * exp(-u)
    i1 = bisect(energy0, e)
    i0 = i1 - 1
    xs_tot = np.zeros(n)
    xs_sca = np.zeros(n)
    if has_res_fis:
        xs_fis = np.zeros(n)
    for i in range(n-1, -1, -1):
        r0 = (energy0[i1] - e) / (energy0[i1] - energy0[i0])
        r1 = 1.0 - r0
        xs_sca[i] = r0 * xs_sca0[i0] + r1 * xs_sca0[i1]
        xs_tot[i] = r0 * xs_tot0[i0] + r1 * xs_tot0[i1]
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
        return {'xs_tot': xs_tot, 'xs_sca': xs_sca, 'xs_fis': xs_fis}
    else:
        return {'xs_tot': xs_tot, 'xs_sca': xs_sca}


def _interp_erg_temp(xs0, xs1, temp0, temp1, emax, emin, n, temp, has_res_fis):
    rt0 = (temp1 - temp) / (temp1 - temp0)
    rt1 = 1.0 - rt0
    du = log(emax / emin) / n
    energy0 = xs0['energy']
    energy1 = xs1['energy']
    xs_tot0 = xs0['xs_tot']
    xs_tot1 = xs1['xs_tot']
    xs_sca0 = xs0['xs_sca']
    xs_sca1 = xs1['xs_sca']
    if has_res_fis:
        xs_fis0 = xs0['xs_fis']
        xs_fis1 = xs1['xs_fis']

    u = log(emax / emin) - du / 2.0
    e = emax * exp(-u)
    i1 = bisect(energy0, e)
    i0 = i1 - 1
    j1 = bisect(energy1, e)
    j0 = j1 - 1
    xs_tot = np.zeros(n)
    xs_sca = np.zeros(n)
    if has_res_fis:
        xs_fis = np.zeros(n)
    for i in range(n-1, -1, -1):
        r0 = (energy0[i1] - e) / (energy0[i1] - energy0[i0])
        r1 = 1.0 - r0
        xs_tot2 = r0 * xs_tot0[i0] + r1 * xs_tot0[i1]
        xs_sca2 = r0 * xs_sca0[i0] + r1 * xs_sca0[i1]
        if has_res_fis:
            xs_fis2 = r0 * xs_fis0[i0] + r1 * xs_fis0[i1]

        r0 = (energy1[j1] - e) / (energy1[j1] - energy1[j0])
        r1 = 1.0 - r0
        xs_tot3 = r0 * xs_tot1[j0] + r1 * xs_tot1[j1]
        xs_sca3 = r0 * xs_sca1[j0] + r1 * xs_sca1[j1]
        if has_res_fis:
            xs_fis3 = r0 * xs_fis1[j0] + r1 * xs_fis1[j1]

        xs_tot[i] = rt0 * xs_tot2 + rt1 * xs_tot3
        xs_sca[i] = rt0 * xs_sca2 + rt1 * xs_sca3
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
        return {'xs_tot': xs_tot, 'xs_sca': xs_sca, 'xs_fis': xs_fis}
    else:
        return {'xs_tot': xs_tot, 'xs_sca': xs_sca}


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

        # Convert energy to eV
        if energy_unit == 'eV':
            emax = emax_eV_MeV
            emin = emin_eV_MeV
        else:
            emax = emax_eV_MeV * 1e6
            emin = emin_eV_MeV * 1e6

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
            hflib = SDLibrary_interpEnergy(acelib, emax, emin, n, has_res_fis)
            del acelib
        else:
            # Find temperature interpolation index
            i0, i1 = _find_temp_interp_index(kT, kTs)

            # Temperature interpolation and energy point interpolation
            acelib0 = self.get_xs_by_kT_grp(f, nuclide, kT_grps[i0],
                                            has_res_fis)
            acelib1 = self.get_xs_by_kT_grp(f, nuclide, kT_grps[i1],
                                            has_res_fis)
            hflib = SDLibrary_interpEnergyTemperature(
                acelib0, acelib1, kTs[i0], kTs[i1], kT, emax, emin, n,
                has_res_fis)
            del acelib0, acelib1

        f.close()

        return hflib

    def to_rmet21_xs_file(self, fname, nuclide, temp, npft, ni, emax, emin,
                          has_res_fis, find_nearest_temp=True):
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
            acelib = self.get_xs_by_kT_grp(f, nuclide, kT_grps[ii], has_res_fis,
                                           getsdlib=False)
            attrs = {'awr': acelib['awr'], 'potential': acelib['potential']}
            xs = _interp_erg(acelib, emax, emin, npft, has_res_fis)
        else:
            # Find temperature interpolation index
            i0, i1 = _find_temp_interp_index(kT, kTs)

            # Temperature interpolation and energy point interpolation
            acelib0 = self.get_xs_by_kT_grp(f, nuclide, kT_grps[i0],
                                            has_res_fis, getsdlib=False)
            acelib1 = self.get_xs_by_kT_grp(f, nuclide, kT_grps[i1],
                                            has_res_fis, getsdlib=False)
            attrs = {'awr': acelib0['awr'], 'potential': acelib0['potential']}
            xs = _interp_erg_temp(acelib0, acelib1, kTs[i0], kTs[i1], emax,
                                  emin, npft, kT, has_res_fis)

        f.close()

        # Write to XS file
        with FortranFile(fname, 'w') as f:
            if has_res_fis:
                fis_flag = 1
            else:
                fis_flag = 0
            f.write_record(np.array([npft, fis_flag, ni], dtype=np.int32))
            f.write_record(np.array([log(emax / emin) / npft],
                                    dtype=np.float32))
            f.write_record(np.array([emax], dtype=np.float32))
            ii = 0
            for i in range(npft / ni):
                f.write_record(np.float32(xs['xs_tot'][ii:ii+ni] -
                                          xs['xs_sca'][ii:ii+ni]))
                f.write_record(np.float32(xs['xs_sca'][ii:ii+ni]))
                if has_res_fis:
                    f.write_record(np.float32(xs['xs_fis'][ii:ii+ni]))
                ii += ni

        return attrs

    def get_xs_by_kT_grp(self, f, nuclide, kT_grp, has_res_fis, getsdlib=True):
        # Energy points
        energy = f[nuclide]['energy'][kT_grp].value
        n_energy = len(energy)

        # Accumulate cross sections
        xs_tot = np.zeros(n_energy)
        xs_sca = np.zeros(n_energy)
        if has_res_fis:
            xs_fis = np.zeros(n_energy)
        else:
            xs_fis = None
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

        if getsdlib:
            acelib = SDLibrary()
            acelib.setPotential(average_potential(nuclide))
            acelib.setAwr(f[nuclide].attrs['atomic_weight_ratio'])
            acelib.setEnergy(energy)
            acelib.setXsTotal(xs_tot)
            acelib.setXsScatter(xs_sca)
            if has_res_fis:
                acelib.setXsFission(xs_fis)
            return acelib
        else:
            return {'potential': average_potential(nuclide), 'awr':
                    f[nuclide].attrs['atomic_weight_ratio'], 'energy': energy,
                    'xs_tot': xs_tot, 'xs_sca': xs_sca, 'xs_fis': xs_fis}
