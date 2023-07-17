import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpi4py import MPI

from hpc_fluid_dynamics.lbm_utils import *

class LBM:
    """
    Main class for the LBM. 
    """
    def __init__(self, 
                 NX = 250, # number of lattice points in the x direction
                 NY = 250, # number of lattice points in the y direction
                 mode = "circle", # initial condition mode (see lbm_utils.py)
                 omega = 0.1, # relaxation parameter
                 parallel = False, # parallel simulation
                 epsilon = 0.05, # perturbation amplitude
            ):
        
        """
        Initialize the parameters of the LBM.
        """
        self.NX = NX
        self.NY = NY
        self.mode = mode
        self.parallel = parallel
        self.omega = omega
        self.epsilon = epsilon
        self.viscosity = 1/3*(1/omega - 0.5)

        # set the constants
        self.velocity_set = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                                      [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.velocity_set_weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36,
                                              1/36, 1/36, 1/36])
        self.sound_speed = 1 / np.sqrt(3)
        self.wall_velocity = np.array([0.1, 0])
        self.p_in = 0.1
        self.p_out = 0.01
        self.d_p = self.p_out - self.p_in
        self.density_in = (self.p_out + self.d_p) / sound_speed**2
        self.density_out = (self.p_out) / sound_speed**2

        # Initialize the PDF
        self.pdf_9xy = init_pdf(NX, NY, mode, self.epsilon)
        
        # Flags for boundary conditions on each edge
        self.is_boundary = {
            "left": True,
            "right": True,
            "top": True,
            "bottom": True,
        }

        # Initialize the MPI communicator
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

            self.sectsX=sectsX
            self.sectsY=sectsY
            self.nxsub = NX//self.sectsX+2
            self.nysub = NY//self.sectsY+2
            self.cartcomm=self.comm.Create_cart(dims=[sectsX,sectsY],periods=[True,True],reorder=False)
            self.rcoords = self.cartcomm.Get_coords(self.rank)

            # Flags for boundary conditions on each edge
            self.is_boundary = {
                'left': self.rcoords[0] == 0,
                'right': self.rcoords[0] == self.sectsY - 1,
                'top': self.rcoords[1] == self.sectsX - 1,
                'bottom': self.rcoords[1] == 0,
            }

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
        pdf_9xy = self.pdf_9xy
        self.nt = nt

        self.velocities = []
        self.densities = []

        for i in range(self.nt):
            if i % 100 == 0:
                print("step ", i)

            # COMMUNICATE
            if self.parallel:
                pdf_9xy = Communicate(pdf_9xy,self.cartcomm,self.sd,self.is_boundary)

            # MOMENT UPDATE 
            density_xy = calc_density(pdf_9xy)
            local_avg_velocity_xy2 = calc_local_avg_velocity(pdf_9xy,density_xy)

            # EQULIBRIUM 
            equilibrium_pdf_9xy = calc_equilibrium_pdf(density_xy, local_avg_velocity_xy2)
            
            # pressure conditions of left and right walls
            if self.is_boundary["left"] or self.is_boundary["right"]:
                    density_in_x_y = np.ones((self.NX, self.NY))*self.density_in
                    density_out_x_y = np.ones((self.NX, self.NY))*self.density_out

                    u1_x_y_2 = np.repeat(local_avg_velocity_xy2[1,:,:][None, :], self.NX, axis=0)
                    uN_x_y_2 = np.repeat(local_avg_velocity_xy2[-2,:,:][None, :], self.NX, axis=0)

                    eq_pdf_u1 = calc_equilibrium_pdf(density_out_x_y, u1_x_y_2)[:, 1, :]
                    eq_pdf_uN = calc_equilibrium_pdf(density_in_x_y, uN_x_y_2)[:, -2, :]
                    
                    self.fill1 = eq_pdf_uN + (pdf_9xy[:, -2, :] - equilibrium_pdf_9xy[:, -2, :]) # x N
                    self.fill2 = eq_pdf_u1 + (pdf_9xy[:, 1, :] - equilibrium_pdf_9xy[:, 1, :]) # x 1

                    # pressure conditions of left and right walls
                    if self.is_boundary["left"]:
                        pdf_9xy[[1, 5, 8],0,:] = self.fill1[[1, 5, 8]]
                    if self.is_boundary["right"]:
                        pdf_9xy[[1, 5, 8],-1,:] = self.fill2[[1, 5, 8]]
                        
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

                    self.densities.append(density_xy_gathered)
                    self.velocities.append(velocity_xy_gathered)

            
            else:
                # SAVE RESULTS
                self.densities.append(density_xy)
                self.velocities.append(local_avg_velocity_xy2)

        self.pdf_9xy = pdf_9xy

         # compute the empirical viscosity in the shear wave context
        if not self.parallel and self.mode == "shear_wave":
            self.amplitudes = []
            for velocity in self.velocities:
                v_norm = np.linalg.norm(velocity, axis=2)
                self.amplitudes.append(np.max(v_norm)-np.min(v_norm))
                viscosities = [viscosity_from_amplitude(t, amplitude, self.epsilon, self.NX) for t, amplitude in enumerate(self.amplitudes)]
                viscosities = viscosities[10:]

            self.measured_viscosity = np.mean(viscosities)


    def boundary_conditions(self,pdf_9xy,density_xy):
        """
        Boundary conditions in the couette, poiseuille and sliding lid contexts
        """
        if self.mode in ['couette', 'lid']:
            opposite_indexes = [[6, 8], [2, 4], [5, 7]] # indexes of opposite directions
            # bounce back conditions on the lower wall
            if self.is_boundary["bottom"]:
                for oi in opposite_indexes:
                    pdf_9xy[oi[0], :, 0] = pdf_9xy[oi[1], :, 0]

            # bounce back conditions on the upper wall (velocity (u,0))
            if self.is_boundary["top"]:
                for oi in opposite_indexes:
                    pdf_9xy[oi[1], :, -1] = pdf_9xy[oi[0], :, -1] - \
                                                    2 * self.velocity_set_weights[oi[0]] * density_xy[:, -1] * np.dot(self.velocity_set[oi[0]], self.wall_velocity) / self.sound_speed**2
            
        if self.mode == 'lid':
            opposite_indexes = [[5, 7], [1, 3], [8, 6]] # indexes of opposite directions
            # bounce back conditions on the left wall
            if self.is_boundary["left"]:
                for oi in opposite_indexes:
                    pdf_9xy[oi[0], 0, :] = pdf_9xy[oi[1], 0, :]
            # bounce back conditions on the right wall 
            if self.is_boundary["right"]:
                for oi in opposite_indexes:
                    pdf_9xy[oi[1], -1, :] = pdf_9xy[oi[0], -1, :]

        if self.mode == "poiseuille":

            opposite_indexes = [[6, 8], [2, 4], [5, 7]] # indexes of opposite directions
            # bounce back conditions on the lower wall
            if self.is_boundary["bottom"]:
                for oi in opposite_indexes:
                    pdf_9xy[oi[0], :, 0] = pdf_9xy[oi[1], :, 0]

            # bounce back conditions on the upper wall 
            if self.is_boundary["top"]:
                for oi in opposite_indexes:
                    pdf_9xy[oi[1], :, -1] = pdf_9xy[oi[0], :, -1]



        return pdf_9xy
    
    """
    Plotting functions
    """

    def animate_density(self,i):
        """
        animating helper for the density.
        """
        self.im.set_array(self.densities[i])
        self.ax.set_title('t = %d' % i)
        return self.im,

    def plot_density(self,filename = "density.gif"):
        """
        Plot the density.
        """
        print("Saving density into an animated graph ...")
        
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

            ## velocity profile at the middle of the domain, every 100 time steps
        for i in range(0, self.nt, 100):
            plt.plot(self.densities[i][self.NX//2,:], label="t = "+str(i))
            plt.title("density profile at the middle of the domain (x = NX/2)")
            plt.xlabel("y")
            plt.ylabel("denstiy")
            #plt.ylim(-0.05,0.05)
            plt.legend()
        plt.savefig("results/density_profile_"+self.mode+".png")
        plt.clf()

    def animate_velocity(self,i):
        """
        animating helper for the velocity in the x and y directions.
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
        print("Saving velocity into an animated graph ...")
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.im1 = self.ax1.imshow(self.velocities[0][:,:,0], cmap='jet',)
        self.im2 = self.ax2.imshow(self.velocities[0][:,:,1], cmap='jet',)
        self.fig.colorbar(self.im1)
        self.fig.colorbar(self.im2)
        anim = animation.FuncAnimation(self.fig, 
                                    self.animate_velocity, 
                                    frames=self.nt)
        anim.save("results/"+filename, 
                writer = 'pillow', 
                fps = 30)
        plt.clf()
        
        ## velocity profile at the middle of the domain, every 100 time steps
        for i in [self.nt//2, self.nt-1]:
            plt.plot(self.velocities[i][self.NX//2,:,0], label="t = "+str(i))
            plt.title("Velocity profile at the middle of the domain (x = NX/2)")
            
        if self.mode == "poiseuille":
            analytical_solution = [self.d_p*y*(y - self.NY)/(2*self.viscosity*calc_velocity(self.pdf_9xy).mean(axis=2).sum()) for y in range(self.NY)]
            plt.plot(analytical_solution, label="analytical velocity")
            
        plt.xlabel("y")
        plt.ylabel("x-component of velocity")
        plt.legend()
        plt.savefig("results/velocity_profile_"+self.mode+".png")
        plt.clf()

        if self.mode == "shear_wave":
            plt.plot(analytic_amplitude(np.arange(self.nt), self.epsilon, self.viscosity, self.NX), label="analytic amplitude")
            plt.plot(self.amplitudes, label="measured amplitude")
            plt.title("Amplitude of the shear wave over time")
            plt.xlabel("time")
            plt.ylabel("amplitude")
            plt.legend()
            plt.savefig("results/amplitude_"+self.mode+".png")
            plt.clf()

        if self.mode == "lid":
            for i in [self.nt//2, self.nt-1]:
            # stream plot of the velocity at the last time step
                plt.streamplot(np.arange(self.NX), np.arange(self.NY), self.velocities[i][:,:,1], self.velocities[i][:,:,0])
                plt.title("Velocity stream plot at t = "+str(i))
                plt.xlabel("x")
                plt.ylabel("y")
                plt.savefig("results/streamplot_"+self.mode+"_"+str(i)+".png")
                plt.clf()





        