# Author : Pierre Schnizer
"""
The python equivalent of the C example found in the GSL Reference document.

It prints the calculational ouput to stdout. The first column is t, the
second y[0] and the thrid y[1]. Plot it with your favourite programm to see
the output.

"""
import sys
import time
import numpy as np
import pygsl._numobj as numx
from pygsl import odeiv

import os
import spiceypy as spice
print(f'os.getenv("SPICE") = {os.getenv("SPICE")}')
spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))

def func(t, y, mu):
    #f = numx.zeros((2,), ) * 1.0
    f = np.zeros(2,)
    f[0] = y[1]
    f[1] = -y[0] - mu * y[1] * (y[0] ** 2 -1)
    ET = spice.utc2et("2025-12-18T12:28:28")
    spice.spkezr("399", ET, "J2000", "NONE", "301")
    return f

def jac(t, y, mu):
    dfdy = numx.ones((2,2),) * 1. 
    dfdy[0, 0] = 0.0
    dfdy[0, 1] = 1.0
    dfdy[1, 0] = -2.0 * mu * y[0] * y[1] - 1.0
    dfdy[1, 1] = -mu * (y[0]**2 - 1.0)
    dfdt = numx.zeros((2,))
    return dfdy, dfdt
    
def run():

    mu = 10.0
    dimension = 2
    # The different possible steppers for the function
    # Comment your favourite one out to test it.
    #stepper = odeiv.step_rk2
    #stepper = odeiv.step_rk4
    #stepper = odeiv.step_rkf45
    #stepper = odeiv.step_rkck
    stepper = odeiv.step_rk8pd
    #stepper = odeiv.step_rk2imp
    #stepper = odeiv.step_rk4imp
    #stepper = odeiv.step_gear1
    #stepper = odeiv.step_gear2
    #stepper = odeiv.step_bsimp
    
    step = stepper(dimension, func, jac, mu)
    # All above steppers exept the odeiv.step_bsimp (Buerlisch - Stoer Improved
    # method can calculate without an jacobian ...
    step = stepper(dimension, func, args=mu)

    control = odeiv.control_y_new(step, 1e-6, 1e-6)
    evolve  = odeiv.evolve(step, control, dimension)
    
    print(f"# Using stepper {step.name()} with order {step.order()}")
    #print  "# Using Control ", control.name()
    #print "# %9s  %9s %9s  %9s " % ("t", "h", "y[0]", "y[1]")
    hstart = 1
    tstart = 0.0
    t1 = (50.0)
    #t1 = (500000.0,)
    ystart = (1.0, 0.0)
    
    t = tstart
    y = ystart
    stamp = time.time()
    nsteps = 1000
    #nsteps = 10000000
    h = hstart
    for i in range(nsteps):
        if t >= t1:
            break
        t, h, y = evolve.apply(t, t1, h, y)
	#print "  %5d % 10.6f % 10.6f  % 10.6f " %(i, t, y[0], y[1])

    else:
        raise ValueError("Maximum number of steps exceeded!")

    print(f"Needed {time.time() - stamp} seconds")
    print(f"  {t} {h} {y[0]} {y[1]} ")

    stamp = time.time()
    t, h, y = evolve.apply_vector(tstart, t1, hstart, ystart, nsteps=1000, hmax=.1)
    print(f"Needed {time.time() - stamp} seconds")
    print(f"  {t} {h} {y[0, 0]}  {y[0, 1]} ")
	
if __name__ == '__main__':
    run()
