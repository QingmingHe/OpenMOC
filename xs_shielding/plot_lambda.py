#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import h5py
from openmoc.library_ce import is_disappearance
import numpy as np
import matplotlib.pyplot as plt


def plot_lambda(acelib, nuclide, kT_grp):
    # Get ACE library
    with h5py.File(acelib) as f:
        energy = f[nuclide]['energy'][kT_grp].value
        n = len(energy)
        xs_abs = np.zeros(n)
        for reaction in f[nuclide]['reactions']:
            xs = f[nuclide]['reactions'][reaction][kT_grp]['xs']
            mt = int(reaction[9:12])
            i = xs.attrs['threshold_idx'] - 1
            n = xs.shape[0]
            if is_disappearance(mt):
                xs_abs[i:i+n] += xs.value

    # Lambda preserving homogeneous xs
    lambda1 = [
        1.00000000000E+00, 9.88330500000E-01, 1.00000000000E+00,
        9.71728900000E-01, 9.60144600000E-01, 9.45288900000E-01,
        8.81759600000E-01, 8.98695600000E-01, 5.94728400000E-01,
        4.33874400000E-01, 2.07093700000E-01, 6.38040700000E-02,
        6.45154300000E-02, 4.92320400000E-01, 2.50276300000E-02,
        2.00000000000E-01, 1.00000000000E+00, 1.00000000000E+00,
        2.00000000000E-01, 2.00000000000E-01, 1.00000000000E+00,
        1.00000000000E+00, 1.00000000000E+00, 1.00000000000E+00,
        1.00000000000E+00, 1.00000000000E+00, 1.00000000000E+00,
        1.00000000000E+00, 1.00000000000E+00, 1.00000000000E+00,
        1.00000000000E+00, 1.00000000000E+00, 2.00000000000E-01
    ]
    lambda1.append(lambda1[-1])
    lambda1 = np.array(lambda1)[::-1]
    # Lambda preserving heterogeneous xs
    lambda2 = [
        0., 1., 0.84661865, 1., 1., 1., 1.,
        1., 0.90289307, 0.57830811, 0.22967529, 0., 0., 0.,
        0., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.
    ]
    lambda2.append(lambda2[-1])
    lambda2 = np.array(lambda2)[::-1]
    # Lambda preserving flux
    lambda3 = [
        -0.64177, -0.64988, -0.61621, -0.58005, -0.52687, -0.54506, -0.52608,
        -0.53445, -0.38588, -0.31529, -0.23896, -0.11097, 0.084251, 0.294793,
        0.740799, 1.101106, 1.085526, 1.080636, 1.082614, 1.088821, 1.093786,
        1.096282, 1.098916, 1.097939, 1.101422, 1.099721, 1.102488, 1.103890,
        1.105750, 1.106114, 1.109370, 1.114145, 1.123804
    ]
    lambda3.append(lambda3[-1])
    lambda3 = np.array(lambda3)[::-1]

    # Group boundaries
    grp_bnds = [2.4780E+04, 1.5030E+04, 9.1180E+03, 5.5300E+03, 3.5191E+03,
                2.2394E+03, 1.4251E+03, 9.0690E+02, 3.6726E+02, 1.4873E+02,
                7.5501E+01, 4.8052E+01, 2.7700E+01, 1.5968E+01, 9.8770E+00,
                4.0000E+00, 3.3000E+00, 2.6000E+00, 2.1000E+00, 1.5000E+00,
                1.3000E+00, 1.1500E+00, 1.1230E+00, 1.0970E+00, 1.0710E+00,
                1.0450E+00, 1.0200E+00, 9.9600E-01, 9.7200E-01, 9.5000E-01,
                9.1000E-01, 8.5000E-01, 7.8000E-01, 6.2500E-01]
    grp_bnds = np.array(grp_bnds)[::-1]

    # Create figure object and axe object by subplots
    fig, ax1 = plt.subplots()

    lw = 3
    ax1.plot(grp_bnds, lambda1, 'r', ls='steps', label='homogeneous lambda', lw=lw)
    ax1.plot(grp_bnds, lambda2, 'k', ls='steps', label='heterogeneous lambda', lw=lw)
    # ax1.plot(grp_bnds, lambda3, 'y', ls='steps', label='homo flux', lw=lw)
    # Set properties
    ax1.set(xscale="log", yscale="linear", xlabel="energy", ylabel="lambda")
    # Show legend specified by label key word in "plot"
    ax1.legend(loc="upper left")

    # Create another axe
    ax2 = ax1.twinx()
    ax2.plot(energy, xs_abs, label='absorption')
    ax2.set(yscale="log", ylabel="cross sections/barn")
    ax2.legend(loc="upper right")

    plt.xlim(4.0, 9118.0)

    # Set title
    # plt.title("", fontsize="large")
    # Show fig or save fig: show(), savefig(figname) and close()
    plt.savefig('lambda.png')
    # plt.show()

if __name__ == '__main__':
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    nuclide = 'U238'
    acelib = os.path.join(os.path.dirname(cross_sections), '%s.h5' % nuclide)
    kT_grp = '294K'
    plot_lambda(acelib, nuclide, kT_grp)
