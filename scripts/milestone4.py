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
velocity_x_y_2 = np.zeros((L, W, 2))

n_steps = 100
display_anim = True

wall_velocity = np.array([0.1, 0])

pdf_9_x_y = calc_equilibrium_pdf(density_x_y, velocity_x_y_2)

fig = plt.figure()
ax = plt.axes()#xlim=(0, L), ylim=(0, W))
im = ax.imshow(calc_density(pdf_9_x_y), cmap='jet')
fig.colorbar(im)

def animate(i):
    if i % 100 == 0:
        print("i = ", i)
    global pdf_9_x_y
    
    # streaming step
    pdf_streamed_9_x_y = streaming(pdf_9_x_y)
    #print(pdf_streamed_9_x_y.shape)

    assert np.allclose(np.sum(pdf_9_x_y), np.sum(pdf_streamed_9_x_y), atol=1e-3) # check mass conservation
    
    # boundary conditions : 
    # periodic in x direction
    for i in [1, 5, 8]:
        pdf_streamed_9_x_y[i, 0, :] = pdf_streamed_9_x_y[i, L-1, :]

    opposite_indexes = [[6, 8], [2, 4], [5, 7]] # indexes of opposite directions

    # bounce back conditions on the lower wall
    for oi in opposite_indexes:
        pdf_streamed_9_x_y[oi[0], :, 0] = pdf_streamed_9_x_y[oi[1], :, 0]

    # bounce back conditions on the upper wall (velocity (u,0))
    for oi in opposite_indexes:
        pdf_streamed_9_x_y[oi[0], :, W-1] = pdf_streamed_9_x_y[oi[1], :, W-1] - 2 * density_x_y[:, W-1] * np.dot(velocity_set[oi[1]], wall_velocity)/np.dot(velocity_set[oi[1]], velocity_set[oi[1]])


    # recalculate the density (rho)
    density = calc_density(pdf_streamed_9_x_y)

    # calculate local average velocity (u)
    velocity_9_x_y = calc_local_avg_velocity(pdf_streamed_9_x_y)

    # calculate equilibrium pdf
    equilibrium_pdf = calc_equilibrium_pdf(density, velocity_9_x_y)

    
    # collision step
    pdf_collision_9_x_y = pdf_streamed_9_x_y + omega*(equilibrium_pdf - pdf_streamed_9_x_y)

    # calculate the viscosity assuming the Stokes flow condition

    assert np.allclose(np.sum(pdf_streamed_9_x_y), np.sum(pdf_collision_9_x_y), atol=1e-3 )# check mass conservation

    pdf_9_x_y = pdf_collision_9_x_y
    # update the image
    #print(velocity_9_x_y.shape)
    
    im.set_array(velocity_9_x_y[:, :, 0])
    # set colorscale from -1 to 1
    im.set_clim(-0.1, 0.1)
    
    #im.set_array(velocity_9_x_y[0:1, L-1,:])
    ax.set_title('x velocity at t = %d' % i)

    return im,


if display_anim: 
    #plot the animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps)#, interval=200, blit=True)
    anim.save("results/ml4_density_anim.gif", writer = 'pillow', fps = 30)

else: # only run the simulation
    for i in range(n_steps):
        animate(i)

plt.clf() #clear the figure