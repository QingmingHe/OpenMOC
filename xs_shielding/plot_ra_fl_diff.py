#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import openmc
import numpy as np

distfile = 'dist/statepoint.1000.h5'
avefile = '975_ave/statepoint.1000.h5'
distsp = openmc.StatePoint(distfile)
avesp = openmc.StatePoint(avefile)
x = [0.0, 0.129653384067, 0.183357574155, 0.224566248577, 0.259306768134,
     0.289913780286, 0.317584634389, 0.343030610879, 0.36671514831,
     0.388960152201, 0.41]


def plot_ce_flux():
    ng = 199
    energy = np.linspace(4.0, 10.0, 200)
    energy = (energy[:199] + energy[1:]) / 2.0
    distflux = distsp.get_tally(id=2).mean.reshape((ng, -1))
    aveflux = avesp.get_tally(id=2).mean.reshape((ng, -1))
    for ireg in [0, 6, 9]:
        plt.plot(energy, distflux[:, ireg], label='nonuniform %i' % (ireg+1),
                 lw=2)
        plt.plot(energy, aveflux[:, ireg], label='uniform %i' % (ireg+1), lw=2)

    plt.yscale('log')
    plt.xlabel('energy (eV)')
    plt.ylabel('flux')
    plt.legend()
    plt.show()


def plot_mg_ra():
    ng = 13
    distflux = distsp.get_tally(id=1).mean[:, 0, 1].reshape((ng, -1))
    distflux = list(distflux[0, :])
    distflux.insert(0, distflux[0])
    distflux = np.array(distflux)
    aveflux = avesp.get_tally(id=1).mean[:, 0, 1].reshape((ng, -1))
    aveflux = list(aveflux[0, :])
    aveflux.insert(0, aveflux[0])
    aveflux = np.array(aveflux)
    fluxerr = (distflux - aveflux) * 100.0 / aveflux

    fig, ax1 = plt.subplots()

    ax1.step(x, distflux, 'k-<', label='non-uniform')
    ax1.step(x, aveflux, 'k->', label='uniform')
    ax1.set(xlabel='radius (cm)', ylabel='absorption reaction rate')
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.step(x, fluxerr, 'k', lw=2, label='difference')
    ax2.set(ylabel='difference (%)', ylim=[-4.0, 4.0])
    ax2.legend(loc="upper right")

    plt.show()


def plot_mg_flux():
    ng = 13
    distflux = distsp.get_tally(id=1).mean[:, 0, 0].reshape((ng, -1))
    distflux = list(distflux[0, :])
    distflux.insert(0, distflux[0])
    distflux = np.array(distflux)
    aveflux = avesp.get_tally(id=1).mean[:, 0, 0].reshape((ng, -1))
    aveflux = list(aveflux[0, :])
    aveflux.insert(0, aveflux[0])
    aveflux = np.array(aveflux)
    fluxerr = (distflux - aveflux) * 100.0 / aveflux

    fig, ax1 = plt.subplots()

    ax1.step(x, distflux, 'k', label='non-uniform')
    ax1.step(x, aveflux, 'r', label='uniform')
    ax1.set(ylim=[0, 0.05], xlabel='radius (cm)', ylabel='flux')
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.step(x, fluxerr, lw=2, label='difference')
    ax2.set(ylabel='difference (%)', ylim=[-1.0, 1.0])
    ax2.legend(loc="upper right")

    plt.show()

plot_mg_ra()
