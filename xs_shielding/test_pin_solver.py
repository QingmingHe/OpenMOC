from resonance_pin_subgroup import PinFixSolver, PINCELLBOX
from time import clock


def test_pinsolver():
    # Test a simple pin problem
    p = PinFixSolver()
    p.pin_type = PINCELLBOX
    p.radii = [0.4095]
    p.pitch = 1.26

    # Test 1
    time0 = clock()
    xs_tot = [1.53097167e+03, 1.35399647e+00]
    xs_sca = [612.38866698, 0.]
    source = [2.13863960e-05, 4.27554870e-04]
    p.set_pin_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
    p.solve()
    print('time for test 1:')
    print(clock() - time0)

    # Test 2
    time0 = clock()
    xs_tot = [0.02887092, 1.35346599]
    xs_sca = [0.01154837, 0.]
    source = [0.05097604, 1.26152386]
    p.set_pin_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
    p.solve()
    print('time for test 2:')
    print(clock() - time0)

    # Test 3. Time is longer than test 1
    time0 = clock()
    xs_tot = [1.53097167e+03, 1.35399647e+00]
    xs_sca = [612.38866698, 0.]
    source = [2.13863960e-05, 4.27554870e-04]
    p.set_pin_xs(xs_tot=xs_tot, xs_sca=xs_sca, source=source)
    p.solve()
    print('time for 2nd run of test 1:')
    print(clock() - time0)
    print('why longer than 1st run of test 1')

if __name__ == '__main__':
    test_pinsolver()
