"""
Couette flow
============================
This script simulates Couette flow, i.e. flow between two parallel plates, 
where the upper plate is moving with a constant velocity.

Initial conditions:
rho(x,y,t=0) = 1
u(x,y,t=0) = 0

Periodic boundary conditions are used in the x-direction.
rho(x,y,t) = rho(x+L,y,t)
u(x,y,t) = u(x+L,y,t)

Lower wall : bounce back
u
"""