"""
Streaming and collision step
============================
The streaming step is the first step of the Lattice Boltzmann Method (LBM).
The streaming step is a simple shift of the pdf in the direction of the
velocity. The collision step is the second step of the LBM. The collision step
is a relaxation of the pdf towards the equilibrium pdf. The equilibrium pdf is
the pdf that would be obtained if the fluid particles were in equilibrium. The
equilibrium pdf is calculated using the density and the velocity of the fluid
particles. 
============================
"""
from matplotlib import animation
from array2gif import write_gif
from hpc_fluid_dynamics.utils import *

omega = 1
n_steps = 500

arrays = []

pdf = init_pdf(mode="circle")

for i in range(n_steps):
    if i % 100 == 0:
        print("step ", i)

    # MOMENT UPDATE 
    density = calc_density(pdf)
    local_avg_velocity = calc_local_avg_velocity(pdf)

    # EQULIBRIUM 
    equilibrium_pdf = calc_equilibrium_pdf(density, local_avg_velocity)

    # COLLISION STEP
    pdf = pdf + omega*(equilibrium_pdf - pdf)

    # STREAMING STEP
    pdf = streaming(pdf)

    # SAVE FOR PLOTTING
    arrays.append(calc_density(pdf))

# PLOT
# add color channel in order to work with array2gif
arrays = np.array(arrays)[..., np.newaxis] * np.ones(3)
# normalize
arrays = arrays / np.max(arrays) * 255
write_gif(arrays, 'results/ml_parallel.gif', fps=30)