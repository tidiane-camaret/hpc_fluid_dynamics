"""
Sheer wave decay
============================
This script simulates the decay of a sheer wave in a 2D lattice.
Same as milestone2.py, but density rho and velocity u are 
intialized to : 
rho = rho0 + epsilon * sin(2*pi*x/L)
u = 0
"""

from matplotlib import animation
from hpc_fluid_dynamics.lbm_utils import *

# quantities
omega = 0.1
rho0 = 0.5
epsilon = 0.05

n_steps = 1000
display_anim = True

L, W = 250, 250
# EX 1 and 2:
distribution = 2 # exercise 1 or 2

incr_array = np.tile(np.arange(L), (L, 1)) # array of increasing integers from 0 to L-1


if distribution == 1:
    #rho = rho0 + epsilon * sin(2*pi*x/L) where x is the x coordinate of the lattice
    density = rho0 + epsilon * np.sin(2*np.pi*incr_array.T/L)
    ### local average velocity (u)
    local_avg_velocity = np.zeros((L, W, 2))

elif distribution == 2:
    density = np.ones((L, W)) * rho0
    ### local average velocity (u)
    local_avg_velocity = np.zeros((L, W, 2))
    local_avg_velocity[:,:,0] = epsilon * np.sin(2*np.pi*incr_array/L)

pdf = calc_equilibrium_pdf(density, local_avg_velocity)

pdf_streamed = streaming(pdf)
equilibrium_pdf = calc_equilibrium_pdf(density, local_avg_velocity)
pdf_collision = pdf_streamed + omega*(equilibrium_pdf - pdf_streamed)
pdf = pdf_collision

fig = plt.figure()
ax = plt.axes(xlim=(0, W), ylim=(0, L))
im = ax.imshow(local_avg_velocity[:,:,0], cmap='jet')
fig.colorbar(im)


amplitudes = []

def animate(i):
    if i % 100 == 0:
        print("i = ", i)
    global pdf
    
    pdf_streamed = streaming(pdf)

    assert np.allclose(np.sum(pdf), np.sum(pdf_streamed), atol=1e-3) # check mass conservation
    
    # recalculate the density (rho)
    density = calc_density(pdf_streamed)

    #print("density = ", density)
    #print("density shape = ", density.shape)
    #print("density average = ", np.average(density))

    # calculate local average velocity (u)
    local_avg_velocity = calc_local_avg_velocity(pdf_streamed)
    local_avg_velocity_norms = np.linalg.norm(local_avg_velocity, axis=2)
    amplitude = np.max(local_avg_velocity_norms) - np.min(local_avg_velocity_norms)
    amplitudes.append(amplitude)
    """
    print("local_avg_velocity = ", local_avg_velocity)
    print("local_avg_velocity shape = ", local_avg_velocity.shape)
    """
    # calculate equilibrium pdf
    equilibrium_pdf = calc_equilibrium_pdf(density, local_avg_velocity)
    """
    print("equilibrium_pdf = ", equilibrium_pdf)
    print("equilibrium_pdf shape = ", equilibrium_pdf.shape)
    """
    
    # collision step
    pdf_collision = pdf_streamed + omega*(equilibrium_pdf - pdf_streamed)
    """
    print("pdf_collision = ", pdf_collision)
    print("pdf_collision shape = ", pdf_collision.shape)
    """

    # calculate the viscosity assuming the Stokes flow condition

    assert np.allclose(np.sum(pdf_streamed), np.sum(pdf_collision), atol=1e-3 )# check mass conservation

    pdf = pdf_collision
    # update the image
    im.set_array(local_avg_velocity[:,:,0])

    ax.set_title('t = %d' % i)

    return im,


if display_anim: 
    #plot the animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps)#, interval=200, blit=True)
    anim.save("results/ml3_velocity_anim_"+str(distribution)+".gif", writer = 'pillow', fps = 30)

else: # only run the simulation
    for i in range(n_steps):
        animate(i)

plt.clf() #clear the figure

print("amplitudes = ", np.array(amplitudes).shape)


# EX 3 :
viscosity = 1/3*(1/omega - 0.5)
# analytic formula for the amplitude
# a(t) = a(0) * exp(-viscosity * t*(2*pi/L)**2 with a(0) = epsilon
def analytic_amplitude(t):
    return epsilon * np.exp(-viscosity * t*(2*np.pi/L)**2)

plt.plot(analytic_amplitude(np.arange(n_steps)), label="analytic amplitude")
plt.plot(amplitudes, label="measured amplitude")
plt.title("Amplitude of the sheer wave over time")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.legend()

plt.savefig("results/ml3_amplitude_"+str(distribution)+".png")