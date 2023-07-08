"""
Sliding lid
============================
This script simulates a sliding lid, i.e. flow between two parallel plates,
where the upper plate is moving with a constant velocity in the x direction
other boundaries have bounce back conditions

Initial conditions:
rho(x,y,t=0) = 1
u(x,y,t=0) = 0

"""

from matplotlib import animation
from hpc_fluid_dynamics.lbm_utils import *

omega = 0.1
n_steps = 1000
wall_velocity = np.array([1, 0])

display_anim = True

fig = plt.figure()
ax = plt.axes()
im = ax.imshow(np.ones((L, W)) , cmap='jet')
fig.colorbar(im)

def animate(i):
    if i % 100 == 0:
        print("i = ", i)
    global pdf_9_x_y
    
    # MOMENT UPDATE 
    if i == 0:
        density_x_y = np.ones((L, W)) 
        velocity_x_y_2 = np.zeros((L, W, 2))
        pdf_9_x_y = calc_equilibrium_pdf(density_x_y, velocity_x_y_2)
    else:
        density_x_y = calc_density(pdf_9_x_y)
        velocity_x_y_2 = calc_local_avg_velocity(pdf_9_x_y)

    # EQULIBRIUM 
    equilibrium_pdf = calc_equilibrium_pdf(density_x_y, velocity_x_y_2)

    # COLLISION STEP
    pdf_9_x_y = pdf_9_x_y + omega*(equilibrium_pdf - pdf_9_x_y)

    # STREAMING STEP
    pdf_9_x_y = streaming(pdf_9_x_y)
    assert np.allclose(np.sum(pdf_9_x_y), np.sum(pdf_9_x_y), atol=1e-3) # check mass conservation
    
    # BOUNDARY CONDITIONS
    opposite_indexes = [[6, 8], [2, 4], [5, 7]] # indexes of opposite directions

    # bounce back conditions on the lower wall
    for oi in opposite_indexes:
        pdf_9_x_y[oi[0], :, 0] = pdf_9_x_y[oi[1], :, 0]

    # bounce back conditions on the upper wall (velocity (u,0))
    for oi in opposite_indexes:
        pdf_9_x_y[oi[1], :, W-1] = pdf_9_x_y[oi[0], :, W-1] - \
                                        2 * velocity_set_weights[oi[0]] * density_x_y[:, W-1] * np.dot(velocity_set[oi[0 ]], wall_velocity) / sound_speed**2

    
    opposite_indexes = [[1, 3], [5, 7], [8, 6]] # indexes of opposite directions
    # bounce back conditions on the left wall
    for oi in opposite_indexes:
        pdf_9_x_y[oi[0], 0, :] = pdf_9_x_y[oi[1], 0, :]

    # bounce back conditions on the right wall 
    for oi in opposite_indexes:
        pdf_9_x_y[oi[1], L-1, :] = pdf_9_x_y[oi[0], L-1, :]
    
    # PLOT X VELOCITY
    im.set_array(velocity_x_y_2[:, :, 0])
    
    im.set_clim(-0.1, 0.1) # set colorscale 
    ax.set_title('x velocity at t = %d' % i)

    return im,


if display_anim: 
    #plot the animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps)#, interval=200, blit=True)
    anim.save("results/ml6_velocity_anim.gif", writer = 'pillow', fps = 30)

else: # only run the simulation
    for i in range(n_steps):
        animate(i)

plt.clf() #clear the figure