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

_TOTAL_XS = 1
_ELASTIC = 2
_N_NONELASTIC = 3
_N_LEVEL = 4
_MISC = 5
_N_2ND = 11
_N_2N = 16
_N_3N = 17
_N_FISSION = 18
_N_F = 19
_N_NF = 20
_N_2NF = 21
_N_NA = 22
_N_N3A = 23
_N_2NA = 24
_N_3NA = 25
_N_NP = 28
_N_N2A = 29
_N_2N2A = 30
_N_ND = 32
_N_NT = 33
_N_N3HE = 34
_N_ND2A = 35
_N_NT2A = 36
_N_4N = 37
_N_3NF = 38
_N_2NP = 41
_N_3NP = 42
_N_N2P = 44
_N_NPA = 45
_N_N1 = 51
_N_N40 = 90
_N_NC = 91
_N_DISAPPEAR = 101
_N_GAMMA = 102
_N_P = 103
_N_D = 104
_N_T = 105
_N_3HE = 106
_N_A = 107
_N_2A = 108
_N_3A = 109
_N_2P = 111
_N_PA = 112
_N_T2A = 113
_N_D2A = 114
_N_PD = 115
_N_PT = 116
_N_DA = 117
_N_5N = 152
_N_6N = 153
_N_2NT = 154
_N_TA = 155
_N_4NP = 156
_N_3ND = 157
_N_NDA = 158
_N_2NPA = 159
_N_7N = 160
_N_8N = 161
_N_5NP = 162
_N_6NP = 163
_N_7NP = 164
_N_4NA = 165
_N_5NA = 166
_N_6NA = 167
_N_7NA = 168
_N_4ND = 169
_N_5ND = 170
_N_6ND = 171
_N_3NT = 172
_N_4NT = 173
_N_5NT = 174
_N_6NT = 175
_N_2N3HE = 176
_N_3N3HE = 177
_N_4N3HE = 178
_N_3N2P = 179
_N_3N3A = 180
_N_3NPA = 181
_N_DT = 182
_N_NPD = 183
_N_NPT = 184
_N_NDT = 185
_N_NP3HE = 186
_N_ND3HE = 187
_N_NT3HE = 188
_N_NTA = 189
_N_2N2P = 190
_N_P3HE = 191
_N_D3HE = 192
_N_3HEA = 193
_N_4N2P = 194
_N_4N2A = 195
_N_4NPA = 196
_N_3P = 197
_N_N3P = 198
_N_3N2PA = 199
_N_5N2P = 200
_N_P0 = 600
_N_PC = 649
_N_D0 = 650
_N_DC = 699
_N_T0 = 700
_N_TC = 749
_N_3HE0 = 750
_N_3HEC = 799
_N_A0 = 800
_N_AC = 849
_N_2N0 = 875
_N_2NC = 891


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
    elif mt in (_N_TA, _N_DT, _N_P3HE, _N_D3HE, _N_3HEA, _N_3P):
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
            acelib = get_xs_by_kT_grp(f, nuclide, kT_grps[ii],
                                      has_res_fis)
            hflib = SDLibrary_interpEnergy(acelib, emax, emin, n, has_res_fis)
            del acelib
        else:
            # Find temperature interpolation index
            i0, i1 = _find_temp_interp_index(kT, kTs)

            # Temperature interpolation and energy point interpolation
            acelib0 = get_xs_by_kT_grp(f, nuclide, kT_grps[i0],
                                       has_res_fis)
            acelib1 = get_xs_by_kT_grp(f, nuclide, kT_grps[i1],
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
            acelib = get_xs_by_kT_grp(f, nuclide, kT_grps[ii], has_res_fis,
                                      getsdlib=False)
            attrs = {'awr': acelib['awr'], 'potential': acelib['potential']}
            xs = _interp_erg(acelib, emax, emin, npft, has_res_fis)
        else:
            # Find temperature interpolation index
            i0, i1 = _find_temp_interp_index(kT, kTs)

            # Temperature interpolation and energy point interpolation
            acelib0 = get_xs_by_kT_grp(f, nuclide, kT_grps[i0],
                                       has_res_fis, getsdlib=False)
            acelib1 = get_xs_by_kT_grp(f, nuclide, kT_grps[i1],
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


def get_xs_by_kT_grp(f, nuclide, kT_grp, has_res_fis, getsdlib=True):
    # Energy points
    energy = f[nuclide]['energy'][kT_grp].value
    n_energy = len(energy)

    # Accumulate cross sections
    xs_tot = np.zeros(n_energy)
    xs_abs = np.zeros(n_energy)
    if has_res_fis:
        xs_fis = np.zeros(n_energy)
    else:
        xs_fis = None
    mts = [int(reaction[9:12]) for reaction in f[nuclide]['reactions']]
    for reaction in f[nuclide]['reactions']:
        mt = int(reaction[9:12])
        # Skip total inelastic level scattering, gas production cross sections
        # (MT=200+), etc.
        if mt == _N_LEVEL or mt == _N_NONELASTIC:
            continue
        if mt > _N_5N2P and mt < _N_P0:
            continue

        # Skip level cross sections if total is available
        if mt >= _N_P0 and mt <= _N_PC and _N_P in mts:
            continue
        if mt >= _N_D0 and mt <= _N_DC and _N_D in mts:
            continue
        if mt >= _N_T0 and mt <= _N_TC and _N_T in mts:
            continue
        if mt >= _N_3HE0 and mt <= _N_3HEC and _N_3HE in mts:
            continue
        if mt >= _N_A0 and mt <= _N_AC and _N_A in mts:
            continue
        if mt >= _N_2N0 and mt <= _N_2NC and _N_2N in mts:
            continue

        xs = f[nuclide]['reactions'][reaction][kT_grp]['xs']
        i = xs.attrs['threshold_idx'] - 1
        n = xs.shape[0]
        xs_tot[i:i+n] += xs.value
        if is_disappearance(mt):
            xs_abs[i:i+n] += xs.value
        if _is_fission(mt):
            xs_abs[i:i+n] += xs.value
            if has_res_fis:
                xs_fis[i:i+n] += xs.value
    xs_sca = xs_tot - xs_abs

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
