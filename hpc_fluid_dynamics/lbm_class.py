import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpi4py import MPI
"""
A class to represent the Lattice Boltzmann Method (LBM).

"""

from hpc_fluid_dynamics.lbm_utils import *

class LBM:
    def __init__(self, 
                 NX = 200, 
                 NY = 250, 
                 mode = "circle", 
                 omega = 0.1,
                 parallel = False,
                 epsilon = 0.05,
            ):
        """
        Initialize the LBM.
        """
        
        self.NX = NX
        self.NY = NY
        self.mode = mode
        self.parallel = parallel

        self.omega = omega
        self.epsilon = epsilon
        self.viscosity = 1/3*(1/omega - 0.5)
        self.velocity_set = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                                      [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.velocity_set_weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36,
                                              1/36, 1/36, 1/36])
        self.sound_speed = 1 / np.sqrt(3)
        self.wall_velocity = np.array([0.1, 0])
        self.pdf_9xy = init_pdf(NX, NY, mode)
        

        if parallel:
            
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
 
            ### Domain decomposition
            if NX < NY:
                sectsX=int(np.floor(np.sqrt(self.size*NX/NY)))
                sectsY=int(np.floor(self.size/sectsX))
                print('We have {} fields in x-direction and {} in y-direction'.format(sectsX,sectsY))
                print('How do the fractions look like?')
                print('NX/NY={} and sectsX/sectsY = {}\n'.format(NX/NY,sectsX/sectsY))
            elif NX > NY:
                sectsY=int(np.floor(np.sqrt(self.size*NY/NX)))
                sectsX=int(np.floor(self.size/sectsY))
                print('We have {} fields in x-direction and {} in y-direction'.format(sectsX,sectsY))
                print('How do the fractions look like?')
                print('NX/NY={} and sectsX/sectsY = {}\n'.format(NX/NY,sectsX/sectsY))
            elif NX==NY:
                sectsY=int(np.floor(np.sqrt(self.size)))
                sectsX=int(self.size/sectsY)
                if self.rank == 0: print('In the case of equal size we divide the processes as {} and {}'.format(sectsX,sectsY))

            #sectsX=int(np.floor(np.sqrt(self.size)))
            #sectsY=int(self.size//sectsX)

            self.sectsX=sectsX
            self.sectsY=sectsY
            self.nxsub = NX//self.sectsX+2
            self.nysub = NY//self.sectsY+2
            self.boundary_k=[False,False,False,False]

            self.cartcomm=self.comm.Create_cart(dims=[sectsX,sectsY],periods=[True,True],reorder=False)
            self.rcoords = self.cartcomm.Get_coords(self.rank)

            # where to receive from and where send to 
            sR,dR = self.cartcomm.Shift(1,1)
            sL,dL = self.cartcomm.Shift(1,-1)

            sU,dU = self.cartcomm.Shift(0,-1)
            sD,dD = self.cartcomm.Shift(0,1)

            self.sd = np.array([sR,dR,sL,dL,sU,dU,sD,dD], dtype = int)

            allrcoords = self.comm.gather(self.rcoords,root = 0)
            allDestSourBuf = np.zeros(self.size*8, dtype = int)
            self.comm.Gather(self.sd, allDestSourBuf, root = 0)

            if self.rank == 0: 
                self.density_plot_list = []
                print(allrcoords)
                print(' ')
                cartarray = np.ones((sectsY,sectsX),dtype=int)
                allDestSour = np.array(allDestSourBuf).reshape((self.size,8))
                for i in np.arange(self.size):
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

            # separate the pdf into subdomains
            self.pdf_9xy = self.pdf_9xy[:,self.rcoords[0]*NY//sectsY:(self.rcoords[0]+1)*NY//sectsY,self.rcoords[1]*NX//sectsX:(self.rcoords[1]+1)*NX//sectsX]



    def run(self, nt = 1000):
        """
        Run the LBM and store the density and velocity at each time step
        """
        self.velocities = []
        self.densities = []
        pdf_9xy = self.pdf_9xy
        self.nt = nt
        for i in range(self.nt):
            if i % 100 == 0:
                print("step ", i)

            # COMMUNICATE
            if self.parallel:
                pdf_9xy = Communicate(pdf_9xy,self.cartcomm,self.sd)

            # MOMENT UPDATE 
            density_xy = calc_density(pdf_9xy)
            local_avg_velocity_xy2 = calc_local_avg_velocity(pdf_9xy,density_xy)

            # EQULIBRIUM 
            equilibrium_pdf_9xy = calc_equilibrium_pdf(density_xy, local_avg_velocity_xy2)

            # COLLISION STEP
            pdf_9xy = pdf_9xy + self.omega*(equilibrium_pdf_9xy - pdf_9xy)

            # STREAMING STEP
            pdf_9xy = streaming(pdf_9xy)

            # BOUNDARY CONDITIONS
            pdf_9xy = self.boundary_conditions(pdf_9xy, density_xy)


            if self.parallel:
                # GATHER AND SAVE RESULTS
                density_1D = np.zeros((self.NX*self.NY)) # 1D array to store density
                velocity_1D = np.zeros((self.NX*self.NY,2)) # 1D array to store velocity
                self.comm.Gather(density_xy.reshape((self.nxsub-2)*(self.nysub-2)), density_1D, root = 0)
                self.comm.Gather(local_avg_velocity_xy2.reshape((self.nxsub-2)*(self.nysub-2),2), velocity_1D, root = 0)

                rcoords_x = self.comm.gather(self.rcoords[1], root=0)
                rcoords_y = self.comm.gather(self.rcoords[0], root=0)
                if self.rank == 0:

                    xy = np.array([rcoords_x,rcoords_y]).T
                    density_xy_gathered = np.zeros((self.NX,self.NY))
                    velocity_xy_gathered = np.zeros((self.NX,self.NY,2))
                    #
                    for i in np.arange(self.sectsX):
                        for j in np.arange(self.sectsY):
                            k = i*self.sectsX+j
                            xlo = self.NX//self.sectsX*xy[k,1]
                            xhi = self.NX//self.sectsX*(xy[k,1]+1)
                            ylo = self.NY//self.sectsY*xy[k,0]
                            yhi = self.NY//self.sectsY*(xy[k,0]+1)
                            clo = k*self.NX*self.NY//(self.sectsX*self.sectsY)
                            chi = (k+1)*self.NX*self.NY//(self.sectsX*self.sectsY)

                            density_xy_gathered[xlo:xhi,ylo:yhi] = density_1D[clo:chi].reshape(self.NX//self.sectsX,self.NY//self.sectsY)
                            velocity_xy_gathered[xlo:xhi,ylo:yhi,:] = velocity_1D[clo:chi,:].reshape(self.NX//self.sectsX,self.NY//self.sectsY,2)
                    #print the middle of the grid
                    #print(density_plot[self.NX//2,self.NY//2])
                    self.densities.append(density_xy_gathered)
                    self.velocities.append(velocity_xy_gathered)

            
            else:
                # SAVE RESULTS
                self.densities.append(density_xy)
                self.velocities.append(local_avg_velocity_xy2)


    def animate_density(self,i):
        """
        Animate the density.
        """
        self.im.set_array(self.densities[i])
        self.ax.set_title('t = %d' % i)
        return self.im,

    def plot_density(self,filename = "density.gif"):
        """
        Plot the density.
        """
        if self.parallel==False or self.rank==0:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.densities[0], cmap='jet')
            self.fig.colorbar(self.im)
            anim = animation.FuncAnimation(self.fig, 
                                        self.animate_density, 
                                        frames=self.nt)
            anim.save("results/"+filename, 
                    writer = 'pillow', 
                    fps = 30)
            
            plt.clf()

    def animate_velocity(self,i):
        """
        animate the velocity in the x and y directions.
        We need 2 graphs for this.
        """
        self.im1.set_array(self.velocities[i][:,:,0])
        self.im2.set_array(self.velocities[i][:,:,1])
        self.ax1.set_title('t = %d' % i)
        return self.im1, self.im2,

    def plot_velocity(self,filename = "velocity.gif"):
        """
        Plot the velocity.
        """
        if self.parallel==False or self.rank==0:

            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
            self.im1 = self.ax1.imshow(self.velocities[0][:,:,0], cmap='jet')
            self.im2 = self.ax2.imshow(self.velocities[0][:,:,1], cmap='jet')
            self.fig.colorbar(self.im1)
            self.fig.colorbar(self.im2)
            anim = animation.FuncAnimation(self.fig, 
                                        self.animate_velocity, 
                                        frames=self.nt)
            anim.save("results/"+filename, 
                    writer = 'pillow', 
                    fps = 30)
            plt.clf()

            if self.mode == 'shear_wave_2':
                amplitudes = []
                for velocity in self.velocities:
                    v_norm = np.linalg.norm(velocity, axis=2)
                    amplitudes.append(np.max(v_norm)-np.min(v_norm))


            
                plt.plot(analytic_amplitude(np.arange(self.nt), self.epsilon, self.viscosity, self.NX), label="analytic amplitude")
                plt.plot(amplitudes, label="measured amplitude")
                plt.title("Amplitude of the sheer wave over time")
                plt.xlabel("time")
                plt.ylabel("amplitude")
                plt.legend()
                plt.savefig("results/amplitude_"+self.mode+".png")
                plt.clf()

    def boundary_conditions(self,pdf_9xy,density_xy):
        if self.mode in ['couette', 'lid']:
            opposite_indexes = [[6, 8], [2, 4], [5, 7]] # indexes of opposite directions
            # bounce back conditions on the lower wall
            for oi in opposite_indexes:
                pdf_9xy[oi[0], :, 0] = pdf_9xy[oi[1], :, 0]

            # bounce back conditions on the upper wall (velocity (u,0))
            for oi in opposite_indexes:
                pdf_9xy[oi[1], :, -1] = pdf_9xy[oi[0], :, -1] - \
                                                2 * self.velocity_set_weights[oi[0]] * density_xy[:, -1] * np.dot(self.velocity_set[oi[0]], self.wall_velocity) / self.sound_speed**2
        if self.mode == 'lid':
            ### TODO moves even with zero velocity. See why.
            opposite_indexes = [[5, 7], [1, 3], [8, 6]] # indexes of opposite directions
            # bounce back conditions on the left wall
            for oi in opposite_indexes:
                pdf_9xy[oi[0], 0, :] = pdf_9xy[oi[1], 0, :]
            # bounce back conditions on the right wall 
            for oi in opposite_indexes:
                pdf_9xy[oi[1], -1, :] = pdf_9xy[oi[0], -1, :]

        return pdf_9xy
    
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

def analytic_amplitude(t, epsilon, viscosity, NX):
    return epsilon * np.exp(-viscosity * t*(2*np.pi/NX)**2)
"""    
def animate_velocity(self,i):

        Here, we have to use quiver instead of imshow.

        self.im.set_UVC(self.velocities[i][:,:,0], self.velocities[i][:,:,1])
        self.ax.set_title('t = %d' % i)

        return self.im,

    def plot_velocity(self,):

        Plot the velocity.

        self.velocities = np.array(self.velocities)
        # normalize the velocity
        self.velocities = self.velocities / (np.max(self.velocities) * 100)
        X, Y = np.meshgrid(np.arange(self.NX), np.arange(self.NY))
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.quiver(X, Y, self.velocities[0][:,:,0], self.velocities[0][:,:,1], pivot='mid', color='r', units='inches')
        self.fig.colorbar(self.im)
        anim = animation.FuncAnimation(self.fig, self.animate_velocity, frames=self.nt)
        anim.save("results/velocity.gif", writer = 'pillow', fps = 30)

"""