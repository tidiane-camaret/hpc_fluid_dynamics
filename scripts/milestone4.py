"""
Couette flow
============================
This script simulates Couette flow, i.e. flow between two parallel plates, 
where the upper plate is moving with a constant velocity.

Initial conditions:
rho(x,y,t=0) = 1
u(x,y,t=0) = 0

"""

from matplotlib import animation
from hpc_fluid_dynamics.utils import *

omega = 0.1
density_x_y = np.ones((L, W)) 
velocity_x_y = np.zeros((L, W, 2))

n_steps = 100
display_anim = True

wall_velocity = np.array([0.1, 0])

pdf_d_x_y = calc_equilibrium_pdf(density_x_y, velocity_x_y)

fig = plt.figure()
ax = plt.axes(xlim=(0, W), ylim=(0, L))
im = ax.imshow(calc_density(pdf_d_x_y), cmap='jet')
fig.colorbar(im)

def animate(i):
    if i % 100 == 0:
        print("i = ", i)
    global pdf_d_x_y
    
    # streaming step
    pdf_streamed_d_x_y = streaming(pdf_d_x_y)

    assert np.allclose(np.sum(pdf_d_x_y), np.sum(pdf_streamed_d_x_y), atol=1e-3) # check mass conservation
    
    # boundary conditions : 
    # periodic in x direction
    pdf_streamed_d_x_y[:, 0, :] = pdf_streamed_d_x_y[:, W-1, :]
    # bounce back conditions on the lower wall
    pdf_streamed_d_x_y[2, :, 0] = pdf_streamed_d_x_y[4, :, 0]
    pdf_streamed_d_x_y[5, :, 0] = pdf_streamed_d_x_y[6, :, 0]
    pdf_streamed_d_x_y[7, :, 0] = pdf_streamed_d_x_y[8, :, 0]
    # bounce back conditions on the upper wall (velocity (u,0))
    pdf_streamed_d_x_y[4, :, L-1] = pdf_streamed_d_x_y[2, :, L-1] - 2 * density_x_y[:, L-1] * np.dot(velocity_set[4], wall_velocity)/np.dot(velocity_set[4], velocity_set[4])
    pdf_streamed_d_x_y[6, :, L-1] = pdf_streamed_d_x_y[5, :, L-1] - 2 * density_x_y[:, L-1] * np.dot(velocity_set[6], wall_velocity)/np.dot(velocity_set[6], velocity_set[6])
    pdf_streamed_d_x_y[8, :, L-1] = pdf_streamed_d_x_y[7, :, L-1] - 2 * density_x_y[:, L-1] * np.dot(velocity_set[8], wall_velocity)/np.dot(velocity_set[8], velocity_set[8])
    # recalculate the density (rho)
    density = calc_density(pdf_streamed_d_x_y)

    # calculate local average velocity (u)
    velocity_x_y = calc_local_avg_velocity(pdf_streamed_d_x_y)

    # calculate equilibrium pdf
    equilibrium_pdf = calc_equilibrium_pdf(density, velocity_x_y)

    
    # collision step
    pdf_collision = pdf_streamed_d_x_y + omega*(equilibrium_pdf - pdf_streamed_d_x_y)

    # calculate the viscosity assuming the Stokes flow condition

    assert np.allclose(np.sum(pdf_streamed_d_x_y), np.sum(pdf_collision), atol=1e-3 )# check mass conservation

    pdf_d_x_y = pdf_collision
    # update the image
    im.set_array(calc_density(pdf_d_x_y))

    ax.set_title('t = %d' % i)

    return im,


if display_anim: 
    #plot the animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps)#, interval=200, blit=True)
    anim.save("results/ml4_density_anim.gif", writer = 'pillow', fps = 30)

else: # only run the simulation
    for i in range(n_steps):
        animate(i)

plt.clf() #clear the figure