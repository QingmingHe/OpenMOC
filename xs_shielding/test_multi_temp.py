#!/usr/bin/env python
from resonance_pin_subgroup import (Material, ResonancePinSubgroup,
                                    PinFixSolver, PinCell, PINCELLBOX)
from library_micro import LibraryMicro
from prob_table import adjust_sub_level
from library_pseudo import LibraryPseudo
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

_MICROMGLIB = os.path.join(os.getenv('HOME'),
                           'Dropbox/work/codes/openmoc/micromgxs',
                           'jeff-3.2-wims69e.h5')


def plot_adjust_sub_level():

    nuclide = 'U238'
    temps = [1190.0, 1100.0, 970.0, 890.0, 820.0]
    ave_temp = 975.0
    alltemps = deepcopy(temps)
    alltemps.append(ave_temp)
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    ig = 26
    savefig = 'adjust_level_error_%i.png' % (ig+1)

    # Load micro library
    mglib = LibraryMicro()
    mglib.load_from_h5(_MICROMGLIB)

    pts = []
    rxs = []
    for temp in alltemps:
        plib = LibraryPseudo()
        plib.nuclides = [nuclide]
        plib.densities = [1.0]
        plib.temperature = temp
        plib.cross_sections = cross_sections
        plib.mglib = mglib
        plib.make()
        rxs.append(plib.get_res_tbl(ig, nuclide))
        pts.append(plib.get_subp_one_nuc(ig, nuclide))
    adjust_sub_level(pts, rxs)
    for i, temp in enumerate(temps):
        rx = rxs[i]
        pt = pts[i]
        n_dilution = len(rx['dilutions'])
        error = np.zeros(n_dilution)
        gc_factor = mglib.get_typical_xs(nuclide, temp, ig, 'lambda')['lambda']
        for ib in range(n_dilution):
            sub_flux = pt['sub_wgt'] / (pt['sub_abs'] + gc_factor *
                                        pt['sub_sca'] + rx['dilutions'][ib])
            error[ib] = (np.sum(pt['sub_abs'] * sub_flux) / np.sum(sub_flux) -
                         rx['res_abs'][ib]) / rx['res_abs'][ib] * 100.0
        plt.plot(rx['dilutions'], error, '-o', label='%s K' % temp)
    plt.xlabel('dilution (barn)')
    plt.ylabel('absorption XS errro (%)')
    plt.xscale('log')
    plt.legend()
    plt.savefig(savefig)
    # plt.show()


def test_294():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel = Material(temperature=975.0, nuclides=['U238'],
                    densities=[2.21546e-2], name='fuel')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel, mod]
    pin.radii = [
        # 0.129653384067,
        # 0.183357574155,
        # 0.224566248577,
        # 0.259306768134,
        # 0.289913780286,
        # 0.317584634389,
        # 0.343030610879,
        # 0.36671514831,
        # 0.388960152201,
        0.41
    ]
    # pin.mat_fill = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    pin.mat_fill = [0, 1]
    # sub = ResonancePinSubgroup(first_calc_g=19, last_calc_g=20)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.sph = True
    sub.mod_sph = True
    sub.solve_onenuc_onetemp()
    sub.print_self_shielded_xs(to_h5='simple-pin-294-sph.h5', to_screen=True)
    sub.write_macro_xs('simple-pin-294-sph-macro.h5')


def test_975():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel = Material(temperature=975.0, nuclides=['U238'],
                    densities=[2.21546e-2], name='fuel')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
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
    # sub = ResonancePinSubgroup(first_calc_g=19, last_calc_g=20)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.sph = True
    sub.mod_sph = True
    sub.solve_onenuc_onetemp()
    sub.print_self_shielded_xs(to_h5='simple-pin-975.h5', to_screen=True)
    sub.write_macro_xs('simple-pin-975-macro.h5')


def test_partial_xs_fit_new(ig):
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    fuel_ave = Material(temperature=975.0, nuclides=['U238'],
                        densities=[2.21546e-2], name='fuel_ave')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Define pin cell at average temperature
    pin_ave = PinCell()
    pin_ave.pin_type = PINCELLBOX
    pin_ave.pitch = 1.26
    pin_ave.materials = [fuel_ave, mod]
    pin_ave.radii = [
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
    pin_ave.mat_fill = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    pin_ave.ave_temp = 975.0
    sub = ResonancePinSubgroup(first_calc_g=ig, last_calc_g=ig+1)
    # sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.pin_cell_ave = pin_ave
    sub.partial_xs_preserve_reaction = True
    sub.solve_partial_xs_fit_new()
    sub.print_self_shielded_xs(to_h5='simple-pin-partial-xs-new-%i-xs.h5' % ig,
                               to_screen=False)


def test_adjust_numdens():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    fuel_ave = Material(temperature=975.0, nuclides=['U238'],
                        densities=[2.21546e-2], name='fuel_ave')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # Define pin cell at average temperature
    pin_ave = PinCell()
    pin_ave.pin_type = PINCELLBOX
    pin_ave.pitch = 1.26
    pin_ave.materials = [fuel_ave, mod]
    pin_ave.radii = [
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
    pin_ave.mat_fill = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    pin_ave.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=26, last_calc_g=27)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.pin_cell_ave = pin_ave
    sub.solve_adjust_numdens()
    sub.print_self_shielded_xs(to_h5='simple-pin-adjust-numdens.h5',
                               to_screen=True)


def test_correlation_variant():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=13, last_calc_g=14)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.solve_correlation_variant()
    sub.print_self_shielded_xs(to_h5='simple-pin-correlation-var.h5',
                               to_screen=True)


def test_partial_xs_fit_var():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=15, last_calc_g=16)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.solve_partial_xs_fit_var()
    sub.print_self_shielded_xs(to_h5='simple-pin-partial-xs-fit-var.h5',
                               to_screen=True)


def test_sim_partial_xs():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    fuel_ave = Material(temperature=975.0, nuclides=['U238'],
                        densities=[2.21546e-2], name='fuel_ave')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # Define pin cell at average temperature
    pin_ave = PinCell()
    pin_ave.pin_type = PINCELLBOX
    pin_ave.pitch = 1.26
    pin_ave.materials = [fuel_ave, mod]
    pin_ave.radii = [
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
    pin_ave.mat_fill = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    pin_ave.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=15, last_calc_g=16)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_cell_ave = pin_ave
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.mod_sph = True
    sub.sph = True
    sub.solve_sim_partial_xs()
    sub.print_self_shielded_xs(to_h5='simple-pin-sim-partial.h5',
                               to_screen=True)
    sub.write_macro_xs('simple-pin-sim-partial-macro.h5')


def test_partial_xs_fit():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=15, last_calc_g=16)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.solve_partial_xs_fit()
    sub.print_self_shielded_xs(to_h5='simple-pin-partial-xs-fit.h5',
                               to_screen=True)


def test_correlation():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=26, last_calc_g=27)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.solve_correlation()
    sub.print_self_shielded_xs(to_h5='simple-pin-correlation.h5',
                               to_screen=True)


def test_adjust_level():
    # OpenMC cross sections
    cross_sections = os.getenv('JEFF_CROSS_SECTIONS')
    # Define materials
    fuel0 = Material(temperature=1190.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel0')
    fuel1 = Material(temperature=1140.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel1')
    fuel2 = Material(temperature=1100.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel2')
    fuel3 = Material(temperature=1060.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel3')
    fuel4 = Material(temperature=1010.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel4')
    fuel5 = Material(temperature=970.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel5')
    fuel6 = Material(temperature=930.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel6')
    fuel7 = Material(temperature=890.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel7')
    fuel8 = Material(temperature=860.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel8')
    fuel9 = Material(temperature=820.0, nuclides=['U238'],
                     densities=[2.21546e-2], name='fuel9')
    mod = Material(temperature=600.0, nuclides=['H1'], densities=[0.0662188],
                   name='moderator')
    # Load micro library
    lib = LibraryMicro()
    lib.load_from_h5(_MICROMGLIB)
    # Define a pin cell
    pin = PinCell()
    pin.pin_type = PINCELLBOX
    pin.pitch = 1.26
    pin.materials = [fuel0, fuel1, fuel2, fuel3, fuel4, fuel5, fuel6, fuel7,
                     fuel8, fuel9, mod]
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
    pin.mat_fill = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pin.ave_temp = 975.0
    # sub = ResonancePinSubgroup(first_calc_g=15, last_calc_g=16)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.solve_adjust_sub_level()
    sub.print_self_shielded_xs(to_h5='simple-pin-adjust-level.h5',
                               to_screen=True)


def run_partial_xs_fit_new():
    import sys
    test_partial_xs_fit_new(int(sys.argv[-1]))


if __name__ == '__main__':
    # recommend run with:
    # $ ... -s 0.005 -a 256
    test_294()
    # test_975()
    # test_adjust_level()
    # test_partial_xs_fit()
    # test_partial_xs_fit_new(21)
    # run_partial_xs_fit_new()
    # test_sim_partial_xs()
    # test_partial_xs_fit_var()
    # test_correlation()
    # test_correlation_variant()
    # test_adjust_numdens()
    # plot_adjust_sub_level()
