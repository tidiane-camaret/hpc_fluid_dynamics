import numpy as np

import matplotlib.pyplot as plt

L, W = 50, 30 # width and length of the grid
velocity_set = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [-1, 1], [-1, -1], [1, -1]])

velocity_set_weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36,
                                    1/36, 1/36])

sound_speed = 1 / np.sqrt(3)

def init_pdf(mode = "random_uniform"): 
    """
    Initialize the probability distribution function (pdf) of the fluid
    particles. The pdf is a 3D array of shape (len(velocity_set), L, W), where
    L and W are the length and width of the grid, respectively.
    We need to avoid zones of zero density, otherwise the simulation will
    crash. 
    """

    pdf = np.ones((len(velocity_set), L, W)) / len(velocity_set)
    if mode == "random_uniform":
        for i in range(len(velocity_set)):
            pdf[i] += np.random.uniform(0, 1, (L, W))

    elif mode == "line":
        pdf[:, :, W//2] += 0.5

    elif mode == "circle":
        for i in range(len(velocity_set)):
            for x in range(L):
                for y in range(W):
                    if (x - L//2)**2 + (y - W//2)**2 < 10**2:
                        pdf[i, x, y] += 0.5

    elif mode == "square":
        for i in range(len(velocity_set)):
            for x in range(L):
                for y in range(W):
                    if abs(x - L//2) < 10 and abs(y - W//2) < 10:
                        pdf[i, x, y] += 0.001

    elif mode == "zero channel":
        for x in range(L):
            for y in range(W):
                if abs(x - L//2) < 10 and abs(y - W//2) < 10:
                    pdf[0, x, y] += 0.5

    else:
        raise ValueError("Invalid mode")
    pdf /= np.sum(pdf)
    return pdf

def calc_density(pdf):
    """
    Calculate the density of the fluid particles. The density is a 2D array of
    shape (L, W), where L and W are the length and width of the grid,
    respectively.
    """
    density = np.sum(pdf, axis=0)
    return density

def calc_velocity(pdf):
    """
    Calculate the velocity of the fluid particles. The velocity is a 3D array
    of shape (L, W, 2), where L and W are the length and width of the grid,
    respectively. The third dimension is the velocity in the x and y
    directions.
    """
    velocity = np.zeros((L, W, 2))
    # loop through all positions
    for i in range(L):
        for j in range(W):
            # loop through all velocities
            for k in range(len(velocity_set)):
                velocity[i, j, 0] += pdf[k, i, j] * velocity_set[k, 0]
                velocity[i, j, 1] += pdf[k, i, j] * velocity_set[k, 1]
            velocity[i, j, 0] /= calc_density(pdf)[i, j]
            velocity[i, j, 1] /= calc_density(pdf)[i, j]
    return velocity

def calc_local_avg_velocity(pdf):
    """
    Calculate the local average velocity of the fluid particles. The local
    average velocity is a 3D array of shape (L, W, 2), where L and W are the
    length and width of the grid, respectively. The third dimension is the
    velocity in the x and y directions.
    """
    density = calc_density(pdf)
    sum = np.zeros((L, W, 2))
    for i in range(len(velocity_set)):
        sum[:, :, 0] += velocity_set[i, 0, None]*pdf[i, :, :]
        sum[:, :, 1] += velocity_set[i, 1, None]*pdf[i, :, :]
    return sum / density[:, :, None] 


def calc_equilibrium_pdf(density, velocity):
    """
    Calculate the equilibrium pdf of the fluid particles. The equilibrium pdf
    is a 3D array of shape (len(velocity_set), L, W), where L and W are the
    length and width of the grid, respectively.
    """
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    equilibrium_pdf = np.zeros((len(velocity_set), L, W))
    for i in range(len(velocity_set)):
        equilibrium_pdf[i] = w[i]*density*\
            (1 + 
             3*(velocity_set[i, 0]*velocity[:, :, 0] + velocity_set[i, 1]*velocity[:, :, 1]) +
             9/2*(velocity_set[i, 0]*velocity[:, :, 0] + velocity_set[i, 1]*velocity[:, :, 1])**2 - 
             3/2*(velocity[:, :, 0]**2 + velocity[:, :, 1]**2))
    return equilibrium_pdf


def streaming(pdf):
    """ 
    Streaming step of the Lattice Boltzmann Method (LBM). The streaming step
    moves the fluid particles according to their velocities. The pdf is a 3D
    array of shape (len(velocity_set), L, W), where L and W are the length and
    width of the grid, respectively.
    """
    pdf_t1 = np.zeros_like(pdf)
    # use np.roll to shift the pdf
    for i in range(len(velocity_set)):
        pdf_t1[i] = np.roll(pdf[i], velocity_set[i], axis=(0,1))
    return pdf_t1


# visualize the pdf
def plot_pdf(pdf):
    plt.figure()
    plt.imshow(calc_density(pdf))
    plt.show()

# visualize the velocity
def plot_velocity(pdf):
    plt.figure()
    velocity = calc_velocity(pdf)
    print(velocity.shape)
    plt.streamplot(np.arange(W), np.arange(L), velocity[:, :, 0], velocity[:, :, 1])
    plt.show()