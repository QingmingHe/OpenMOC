#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" brief description

Author: Qingming He
Email: he_qing_ming@foxmail.com
Date: 2016-10-28 14:43:17

Description
===========


"""
import os
from resonance_pin_subgroup import (Material, PinCell, PINCELLBOX,
                                    ResonancePinSubgroup, PinFixSolver)
from library_micro import LibraryMicro
import numpy as np


def _solve_pin(res_nuc, ig, pin, lib, gc_factor, cross_sections):
    # Reset Goldstein-Cohen factor in multi-group library
    lib.set_gc_factor(res_nuc, ig, gc_factor)

    # Solve the problem
    sub_solver = ResonancePinSubgroup(first_calc_g=ig, last_calc_g=ig+1)
    sub_solver.pin_cell = pin
    sub_solver.pin_solver = PinFixSolver()
    sub_solver.pin_solver.n_ring_fuel = 20
    sub_solver.cross_sections = cross_sections
    sub_solver.use_pseudo_lib = True
    sub_solver.micro_lib = lib
    sub_solver.solve_onenuc_onetemp()

    # Return the self-shielded cross section
    jg = ig - lib.first_res
    xs = sub_solver.resnuc_xs
    return xs[res_nuc]['xs_abs'][jg, sub_solver.n_res_reg]


def search_lambda():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    res_nuc = 'U238'
    fuel = Material(temperature=975.0, nuclides=[res_nuc],
                    densities=[2.21546e-2], name='fuel')
    mod = Material(temperature=600.0, nuclides=['H1'],
                   densities=[0.0662188], name='moderator')

    # Load micro library
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmc/openmc/micromgxs',
                         'jeff-3.2-wims69e.h5')
    lib = LibraryMicro()
    lib.load_from_h5(fname)

    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel, mod]
    pin.radii = [
        0.129653384067,
        0.183357574155,
        0.224566248577,
        0.259306768134,
        0.289913780286,
        0.317584634389,
        0.343030610879,
        0.36671514831,
        0.388960152201,
        0.41
    ]
    pin.mat_fill = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # The reference self-shielded xs
    ref_xs_abs = np.array([
        0.5226338498, 0.6272999690, 0.7310751679, 0.7864303188, 1.0500013692,
        1.0137941643, 1.4013544272, 1.6713807513, 1.9093500475, 2.8526174370,
        2.3513961476, 4.5904784947, 6.3146473462, 0.4743688808, 8.1354607075,
        0.6380476135, 0.5137847193, 0.4648261301, 0.4542975535, 0.4655661499,
        0.4786617039, 0.4871682954, 0.4901645764, 0.4933614742, 0.4966692551,
        0.5002436474, 0.5037097883, 0.5074206382, 0.5111444645, 0.5163989932,
        0.5258026157, 0.5396899828, 0.5703013368
    ])

    # Search lambda for each group
    gc_factors = np.zeros(lib.last_res - lib.first_res)
    for ig in range(lib.first_res, lib.last_res):
        print("searching group %i ..." % (ig))
        jg = ig - lib.first_res
        min_xs_abs = _solve_pin(res_nuc, ig, pin, lib, 0.0, cross_sections)
        max_xs_abs = _solve_pin(res_nuc, ig, pin, lib, 1.0, cross_sections)
        if ref_xs_abs[jg] <= min_xs_abs:
            gc_factors[jg] = 0.0
        elif ref_xs_abs[jg] >= max_xs_abs:
            gc_factors[jg] = 1.0
        else:
            left = 0.0
            right = 1.0
            while True:
                if right - left < 1e-4:
                    break
                mid = (left + right) / 2.0
                xs_abs = _solve_pin(res_nuc, ig, pin, lib, mid, cross_sections)
                if xs_abs >= ref_xs_abs[jg]:
                    right = mid
                else:
                    left = mid
            gc_factors[jg] = mid
        print("lambda = %f" % (gc_factors[jg]))

    print gc_factors

if __name__ == '__main__':
    search_lambda()
