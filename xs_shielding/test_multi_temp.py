#!/usr/bin/env python
from resonance_pin_subgroup import (Material, ResonancePinSubgroup,
                                    PinFixSolver, PinCell, PINCELLBOX)
from library_micro import LibraryMicro


def test_one_nuc_multi_temp():
    import os
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
    fname = os.path.join(os.getenv('HOME'),
                         'Dropbox/work/codes/openmc/openmc/micromgxs',
                         'jeff-3.2-wims69e.h5')
    lib = LibraryMicro()
    lib.load_from_h5(fname)
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
    # sub = ResonancePinSubgroup(first_calc_g=23, last_calc_g=24)
    sub = ResonancePinSubgroup()
    sub.pin_cell = pin
    sub.pin_solver = PinFixSolver()
    sub.pin_solver.n_ring_fuel = 20
    sub.micro_lib = lib
    sub.cross_sections = cross_sections
    sub.use_pseudo_lib = True
    sub.solve_adjust_sub_wgt()
    sub.print_self_shielded_xs(to_h5='simple-pin.h5', to_screen=True)


if __name__ == '__main__':
    # recommend run with:
    # $ ... -s 0.005 -a 256
    test_one_nuc_multi_temp()
