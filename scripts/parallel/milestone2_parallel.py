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

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from array2gif import write_gif

from hpc_fluid_dynamics.lbm_utils import *


comm = MPI.COMM_WORLD      # start the communicator assign to comm
size = comm.Get_size()     # get the size and assign to size
rank = comm.Get_rank()


nt = 1000  # timesteps to iterate
dt = 0.0001   # timestep length

omega = 1
print('Rank/Size {}/{}'.format(rank,size))

NX = 256
NY = 256

### Domain decomposition
if NX < NY:
    sectsX=int(np.floor(np.sqrt(size*NX/NY)))
    sectsY=int(np.floor(size/sectsX))
    print('We have {} fields in x-direction and {} in y-direction'.format(sectsX,sectsY))
    print('How do the fractions look like?')
    print('NX/NY={} and sectsX/sectsY = {}\n'.format(NX/NY,sectsX/sectsY))
elif NX > NY:
    sectsY=int(np.floor(np.sqrt(size*NY/NX)))
    sectsX=int(np.floor(size/sectsY))
    print('We have {} fields in x-direction and {} in y-direction'.format(sectsX,sectsY))
    print('How do the fractions look like?')
    print('NX/NY={} and sectsX/sectsY = {}\n'.format(NX/NY,sectsX/sectsY))
elif NX==NY:
    sectsY=int(np.floor(np.sqrt(size)))
    sectsX=int(size/sectsY)
    if rank == 0: print('In the case of equal size we divide the processes as {} and {}'.format(sectsX,sectsY))

sectsX=int(np.floor(np.sqrt(size)))
sectsY=int(size//sectsX)

nxsub = NX//sectsX+2
nysub = NY//sectsY+2
boundary_k=[False,False,False,False]
cartcomm=comm.Create_cart(dims=[sectsX,sectsY],periods=[True,True],reorder=False)
rcoords = cartcomm.Get_coords(rank)

# where to receive from and where send to 
sR,dR = cartcomm.Shift(1,1)
sL,dL = cartcomm.Shift(1,-1)

sU,dU = cartcomm.Shift(0,-1)
sD,dD = cartcomm.Shift(0,1)

sd = np.array([sR,dR,sL,dL,sU,dU,sD,dD], dtype = int)

allrcoords = comm.gather(rcoords,root = 0)
allDestSourBuf = np.zeros(size*8, dtype = int)
comm.Gather(sd, allDestSourBuf, root = 0)

if rank == 0: 
    density_plot_list = []
    print(allrcoords)
    print(' ')
    cartarray = np.ones((sectsY,sectsX),dtype=int)
    allDestSour = np.array(allDestSourBuf).reshape((size,8))
    for i in np.arange(size):
        cartarray[allrcoords[i][0],allrcoords[i][1]] = i
        print('Rank {} all destinations and sources {}'.format(i,allDestSour[i,:]))
        sR,dR,sL,dL,sU,dU,sD,dD = allDestSour[i]
        print('Rank {} is at {}'.format(i,allrcoords[i]))
        print('sour/dest right {} {}'.format(sR,dR))
        print('sour/dest left  {} {}'.format(sL,dL))  
        print('sour/dest up    {} {}'.format(sU,dU))
        print('sour/dest down  {} {}'.format(sD,dD))
        #print('[stdout:',i,']',allDestSour[i])
    print('')
    print(cartarray)

def Communicate(pdf_9xy,cartcomm,sd):
    recvbuf = np.zeros(pdf_9xy[:,:,1].shape)
    sR,dR,sL,dL,sU,dU,sD,dD = sd
    # Send to right which is destination rigth (dR) and receive from left which is source right (sR)
    # print(rank,'Right, source',sR,'destination',dR)
    sendbuf = pdf_9xy[:,:,-2].copy() # Send the second last column to dR
    cartcomm.Sendrecv(sendbuf, dR, recvbuf = recvbuf, source = sR)
    pdf_9xy[:,:,0] = recvbuf # received into the 0th column from sR
    # Send to left and receive from right
    #print(rank,'Left, source',sL,'destination',dL)
    sendbuf = pdf_9xy[:,:,1].copy()
    cartcomm.Sendrecv(sendbuf, dL, recvbuf = recvbuf, source = sL)
    pdf_9xy[:,:,-1] = recvbuf
    # Send to up and receive from down
    #print(rank,'Up, source',sU,'destination',dU)
    sendbuf = pdf_9xy[:,1,:].copy()
    cartcomm.Sendrecv(sendbuf, dU, recvbuf = recvbuf, source = sU)
    pdf_9xy[:,-1,:] = recvbuf
    # Send to down and receive from up
    #print(rank,'Down, source',sD,'destination',dD)
    sendbuf = pdf_9xy[:,-2,:].copy()
    cartcomm.Sendrecv(sendbuf, dD, recvbuf = recvbuf, source = sD)
    pdf_9xy[:,0,:]=recvbuf
#
    return pdf_9xy

## INTIALIZE THE GRID

pdf_9xy_full_range = init_pdf(NX,NY,mode = "square")
pdf_9xy = pdf_9xy_full_range[:,rcoords[0]*NY//sectsY:(rcoords[0]+1)*NY//sectsY,rcoords[1]*NX//sectsX:(rcoords[1]+1)*NX//sectsX]

for t in np.arange(nt):
    # First we need a communication step
    pdf_9xy = Communicate(pdf_9xy,cartcomm,sd)
    # Then we do a timestep forward
    # MOMENT UPDATE 
    density = calc_density(pdf_9xy)
    local_avg_velocity = calc_local_avg_velocity(pdf_9xy)

    # EQULIBRIUM 
    equilibrium_pdf_9xy = calc_equilibrium_pdf(density, local_avg_velocity)

    # COLLISION STEP
    pdf_9xy = pdf_9xy + omega*(equilibrium_pdf_9xy - pdf_9xy)

    # STREAMING STEP
    pdf_9xy = streaming(pdf_9xy)
    density_full_range = np.zeros((NX*NY))

    #comm.Gather(density[1:-1,1:-1].reshape((nxsub-2)*(nysub-2)), density_full_range, root = 0)
    comm.Gather(density.reshape((nxsub-2)*(nysub-2)), density_full_range, root = 0)
    rcoords_x = comm.gather(rcoords[1], root=0)
    rcoords_y = comm.gather(rcoords[0], root=0)
    if rank == 0:

        X0, Y0 = np.meshgrid(np.arange(NX),np.arange(NY))
        xy = np.array([rcoords_x,rcoords_y]).T
        density_plot = np.zeros((NX,NY))
        #
        for i in np.arange(sectsX):
            for j in np.arange(sectsY):
                k = i*sectsX+j
                xlo = NX//sectsX*xy[k,1]
                xhi = NX//sectsX*(xy[k,1]+1)
                ylo = NY//sectsY*xy[k,0]
                yhi = NY//sectsY*(xy[k,0]+1)
                clo = k*NX*NY//(sectsX*sectsY)
                chi = (k+1)*NX*NY//(sectsX*sectsY)

                density_plot[xlo:xhi,ylo:yhi] = density_full_range[clo:chi].reshape(NX//sectsX,NY//sectsY)
        #print the middle of the grid
        print(density_plot[NX//2,NY//2])
        density_plot_list.append(density_plot)

if rank == 0:
    
    c_plot_list = np.array(density_plot_list)
    print(c_plot_list.shape)
    c_plot_list = c_plot_list[..., np.newaxis] * np.ones(3)
    c_plot_list = c_plot_list / np.max(c_plot_list) * 255
    write_gif(c_plot_list, 'results/ml2_parallel_cluster.gif', fps=30)

