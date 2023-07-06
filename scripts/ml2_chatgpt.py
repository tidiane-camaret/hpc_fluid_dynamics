from mpi4py import MPI
import numpy as np
from hpc_fluid_dynamics.utils import *

# Define MPI constants
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define parameters for the lattice Boltzmann simulation
omega = 1
n_steps = 1000

# Initialize the pdf
pdf = init_pdf(mode="circle")
N = 250
# Divide the grid into patches
# Assume that size is a perfect square
grid_size = int(np.sqrt(size))
sub_grid_size = int(N / grid_size)

# Initialize arrays for communication
sendbuf = np.zeros((sub_grid_size, sub_grid_size))
recvbuf = np.zeros((sub_grid_size, sub_grid_size))

# Perform the lattice Boltzmann simulation
for i in range(n_steps):
    # Divide the pdf into patches
    for x in range(grid_size):
        for y in range(grid_size):
            sendbuf = pdf[x*sub_grid_size:(x+1)*sub_grid_size, y*sub_grid_size:(y+1)*sub_grid_size]
            
            # Send patches to corresponding processes
            dest = x * grid_size + y
            comm.Send(sendbuf, dest=dest)

    # Each process receives its patch and performs calculations
    comm.Recv(recvbuf, source=MPI.ANY_SOURCE)
    
    # MOMENT UPDATE 
    density = calc_density(recvbuf)
    local_avg_velocity = calc_local_avg_velocity(recvbuf)

    # EQUILIBRIUM 
    equilibrium_pdf = calc_equilibrium_pdf(density, local_avg_velocity)

    # COLLISION STEP
    recvbuf = recvbuf + omega*(equilibrium_pdf - recvbuf)

    # STREAMING STEP
    recvbuf = streaming(recvbuf)

    # Gather patches back into the full pdf
    for x in range(grid_size):
        for y in range(grid_size):
            source = x * grid_size + y
            comm.Recv(sendbuf, source=source)
            pdf[x*sub_grid_size:(x+1)*sub_grid_size, y*sub_grid_size:(y+1)*sub_grid_size] = sendbuf

# Save the final density field for visualization
np.save('density_field.npy', calc_density(pdf))
