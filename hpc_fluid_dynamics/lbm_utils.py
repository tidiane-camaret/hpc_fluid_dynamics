import numpy as np
import matplotlib.pyplot as plt

velocity_set = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [-1, 1], [-1, -1], [1, -1]])

velocity_set_weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36,
                                    1/36, 1/36])

sound_speed = 1 / np.sqrt(3)

def init_pdf(NX = 250, NY = 250, mode = "random_uniform"): 
    """
    Initialize the probability distribution function (pdf) of the fluid
    particles. The pdf is a 3D array of shape (len(velocity_set), NX, NY), where
    NX and NY are the length and width of the grid, respectively.
    We need to avoid zones of zero density, otherwise the simulation will
    crash. 
    """
    rho0 = 0.5
    epsilon = 0.05
    incr_array = np.tile(np.arange(NX), (NX, 1)) # array of increasing integers from 0 to L-

    pdf = np.ones((len(velocity_set), NX, NY)) / len(velocity_set)
    if mode == "random_uniform":
        for i in range(len(velocity_set)):
            pdf[i] += np.random.uniform(0, 1, (NX, NY))

    elif mode == "line":
        pdf[:, :, NY//2-20:NY//2+20] += 0.5

    elif mode == "circle":
        for i in range(len(velocity_set)):
            for x in range(NX):
                for y in range(NY):
                    if (x - NX//2)**2 + (y - NY//2)**2 < 10**2:
                        pdf[i, x, y] += 0.5

    elif mode == "square":
        for i in range(len(velocity_set)):
            for x in range(NX):
                for y in range(NY):
                    if abs(x - NX//2) < 10 and abs(y - NY//2) < 10:
                        pdf[i, x, y] += 0.5

    elif mode == "zero_channel":
        for x in range(NX):
            for y in range(NY):
                if abs(x - NX//2) < 10 and abs(y - NY//2) < 10:
                    pdf[0, x, y] += 0.5

    elif mode == "shear_wave_1":
        #rho = rho0 + epsilon * sin(2*pi*x/L) where x is the x coordinate of the lattice
        density = rho0 + epsilon * np.sin(2*np.pi*incr_array.T/NX)
        ### local average velocity (u)
        velocity = np.zeros((NX, NY, 2))
        pdf = calc_equilibrium_pdf(density, velocity)

    elif mode == "shear_wave_2":
        density = np.ones((NX, NY)) * rho0
        ### local average velocity (u)
        velocity = np.zeros((NX, NY, 2))
        velocity[:,:,0] = epsilon * np.sin(2*np.pi*incr_array/NX)
        pdf = calc_equilibrium_pdf(density, velocity)

    elif mode in ['couette', 'lid', 'poiseuille']:
        density = np.ones((NX, NY)) 
        velocity = np.zeros((NX, NY, 2))
        pdf = calc_equilibrium_pdf(density, velocity)



    else:
        raise ValueError("Invalid mode")
    pdf /= np.sum(pdf)
    return pdf

def calc_density(pdf):
    """
    Calculate the density of the fluid particles. The density is a 2D array of
    shape (NX, NY), where NX and NY are the length and width of the grid,
    respectively.
    """
    density = np.sum(pdf, axis=0)
    return density

def calc_velocity(pdf):
    """
    Calculate the velocity of the fluid particles. The velocity is a 3D array
    of shape (NX, NY, 2), where NX and NY are the length and width of the grid,
    respectively. The third dimension is the velocity in the x and y
    directions.
    """
    velocity = np.zeros(pdf.shape[1:] + (2,)) 
    # loop through all positions
    for i in range(pdf.shape[1]):
        for j in range(pdf.shape[2]):
            # loop through all velocities
            for k in range(len(velocity_set)):
                velocity[i, j, 0] += pdf[k, i, j] * velocity_set[k, 0]
                velocity[i, j, 1] += pdf[k, i, j] * velocity_set[k, 1]
            velocity[i, j, 0] /= calc_density(pdf)[i, j]
            velocity[i, j, 1] /= calc_density(pdf)[i, j]
    return velocity

def calc_local_avg_velocity(pdf,density):
    """
    Calculate the local average velocity of the fluid particles. The local
    average velocity is a 3D array of shape (NX, NY, 2), where NX and NY are the
    length and width of the grid, respectively. The third dimension is the
    velocity in the x and y directions.
    """
    sum = np.zeros(density.shape + (2,))
    for i in range(len(velocity_set)):
        sum[:, :, 0] += velocity_set[i, 0, None]*pdf[i, :, :]
        sum[:, :, 1] += velocity_set[i, 1, None]*pdf[i, :, :]
    return sum / density[:, :, None] 


def calc_equilibrium_pdf(density, velocity):
    """
    Calculate the equilibrium pdf of the fluid particles. The equilibrium pdf
    is a 3D array of shape (len(velocity_set), NX, NY), where NX and NY are the
    length and width of the grid, respectively.
    """
    equilibrium_pdf = np.zeros((len(velocity_set),) + density.shape)
    for i in range(len(velocity_set)):
        equilibrium_pdf[i] = velocity_set_weights[i] * density * \
            (
                1 + 
                3*(velocity_set[i, 0]*velocity[:, :, 0] + velocity_set[i, 1]*velocity[:, :, 1]) +
                9/2*(velocity_set[i, 0]*velocity[:, :, 0] + velocity_set[i, 1]*velocity[:, :, 1])**2 - 
                3/2*(velocity[:, :, 0]**2 + velocity[:, :, 1]**2)
             )
        
    return equilibrium_pdf


def streaming(pdf):
    """ 
    Streaming step of the Lattice Boltzmann Method (LBM). The streaming step
    moves the fluid particles according to their velocities. The pdf is a 3D
    array of shape (len(velocity_set), NX, NY), where NX and NY are the length and
    width of the grid, respectively.
    """
    pdf_t1 = np.zeros_like(pdf)
    # use np.roll to shift the pdf
    for i in range(len(velocity_set)):
        pdf_t1[i] = np.roll(pdf[i], velocity_set[i], axis=(0,1))
    return pdf_t1

