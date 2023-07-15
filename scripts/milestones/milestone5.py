"""
Poiseuille flow 
============================
This script simulates Poiseuille flow, i.e. flow between two parallel plates,
with a density gradient in the x direction.

Initial conditions:
rho(x,y,t=0) = 1
u(x,y,t=0) = 0

"""

from matplotlib import animation
from hpc_fluid_dynamics.lbm_utils import *
omega = 1.2
p_in = 0.1
p_out = 0.01
d_p = p_out - p_in

density_in = (p_out + d_p) / sound_speed**2
density_out = (p_out) / sound_speed**2

print("density_in = ", density_in)
print("density_out = ", density_out)


n_steps = 500
display_anim = True

NX, NY = 250, 50
fig = plt.figure()
ax = plt.axes()
im = ax.imshow(np.ones((NX, NY)) , cmap='jet')
fig.colorbar(im)

plot_dict = {}

def animate(i):
    if i % 100 == 0:
        print("i = ", i)
    global pdf_9_x_y
    
    # MOMENT UPDATE 
    if i == 0:
        density_x_y = np.ones((NX, NY)) 
        velocity_x_y_2 = np.zeros((NX, NY, 2))
        pdf_9_x_y = calc_equilibrium_pdf(density_x_y, velocity_x_y_2)
    else:
        density_x_y = calc_density(pdf_9_x_y)
        velocity_x_y_2 = calc_local_avg_velocity(pdf_9_x_y, density_x_y)

    # EQULIBRIUM 
    eq_pdf_9_x_y = calc_equilibrium_pdf(density_x_y, velocity_x_y_2)

    
    # PRESSURE BOUNDARY CONDITIONS
    """
    print((np.ones((1, W))*density_in).shape)
    print(velocity_x_y_2[L-2,:,:][None, :].shape)
    print(calc_equilibrium_pdf(np.ones((1, W))*density_in, velocity_x_y_2[L-2,:,:][None, :]).shape)
    print(pdf_9_x_y[:, L-2, :][:, None, :].shape)
    print(equilibrium_pdf[:, L-2, :][:, None, :].shape)
    """
    """
    # at x=0
    pdf_9_x_y[:,0,:] = calc_equilibrium_pdf(np.ones((1, W))*density_in, velocity_x_y_2[L-2,:,:][None, :]).squeeze() + \
                       pdf_9_x_y[:, L-2, :] - \
                       equilibrium_pdf[:, L-2, :]
    

    
    # at x=L-1
    pdf_9_x_y[:,L-1,:] = calc_equilibrium_pdf(np.ones((1, W))*density_out, velocity_x_y_2[1,:,:][None, :]).squeeze() + \
                            pdf_9_x_y[:, 1, :] - \
                            equilibrium_pdf[:, 1, :]
    """

    density_in_x_y = np.ones((NX, NY))*density_in
    density_out_x_y = np.ones((NX, NY))*density_out

    u1_x_y_2 = np.repeat(velocity_x_y_2[1,:,:][None, :], NX, axis=0)
    uN_x_y_2 = np.repeat(velocity_x_y_2[-2,:,:][None, :], NX, axis=0)

    eq_pdf_u1 = calc_equilibrium_pdf(density_out_x_y, u1_x_y_2)[:, 1, :]
    eq_pdf_uN = calc_equilibrium_pdf(density_in_x_y, uN_x_y_2)[:, -2, :]
    
    #print(eq_pdf_u1.shape, pdf_9_x_y[:, -2, :].shape, eq_pdf_9_x_y[:, -2, :].shape)
    fill1 = eq_pdf_uN + (pdf_9_x_y[:, -2, :] - eq_pdf_9_x_y[:, -2, :]) # x N
    fill2 = eq_pdf_u1 + (pdf_9_x_y[:, 1, :] - eq_pdf_9_x_y[:, 1, :]) # x 1

    pdf_9_x_y[[1, 5, 8],0,:] = fill1[[1, 5, 8]]
    pdf_9_x_y[[1, 5, 8], -1,:] = fill2[[1, 5, 8]]
    # COLLISION STEP
    pdf_9_x_y = pdf_9_x_y + omega*(eq_pdf_9_x_y - pdf_9_x_y)

    # STREAMING STEP
    pdf_9_x_y = streaming(pdf_9_x_y)
    assert np.allclose(np.sum(pdf_9_x_y), np.sum(pdf_9_x_y), atol=1e-3) # check mass conservation
    
    # BOUNDARY CONDITIONS
    opposite_indexes = [[6, 8], [2, 4], [5, 7]] # indexes of opposite directions

    # bounce back conditions on the lower wall
    for oi in opposite_indexes:
        pdf_9_x_y[oi[0], :, 0] = pdf_9_x_y[oi[1], :, 0]

    # bounce back conditions on the upper wall
    for oi in opposite_indexes:
        pdf_9_x_y[oi[1], :, NY-1] = pdf_9_x_y[oi[0], :, NY-1] 
        
    # PLOT X VELOCITY
    im.set_array(velocity_x_y_2[:, :, 0])
    #im.set_array(calc_density(pdf_9_x_y))
    im.set_clim(0, 0.2) # set colorscale 
    ax.set_title('x velocity at t = %d' % i)

    if i % 100 == 0:
        plot_dict[i] = velocity_x_y_2[NX//2, :, 0]
   
    return im,


if display_anim: 
    #plot the animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps)#, interval=200, blit=True)
    anim.save("results/ml5_density_anim.gif", writer = 'pillow', fps = 30)
    
else: # only run the simulation
    for i in range(n_steps):
        animate(i)

plt.clf() #clear the figure

for i in plot_dict.keys():
    plt.plot(plot_dict[i], label='t=%d' % i)


viscosity = 1/3*(1/omega - 0.5)
#analytical_solution = [-d_p*y*(y - W)/(2*viscosity*np.ones((L, W)).mean(axis=1).sum()) for y in range(W)]
#plt.plot(analytical_solution, label='analytical')
plt.legend()
plt.savefig("results/ml5_density.png")
