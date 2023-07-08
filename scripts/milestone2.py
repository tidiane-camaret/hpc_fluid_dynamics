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

from matplotlib import animation
from hpc_fluid_dynamics.lbm_utils import *

pdf = init_pdf(mode="circle")

fig = plt.figure()
ax = plt.axes(xlim=(0, W), ylim=(0, L))
im = ax.imshow(calc_density(pdf), cmap='jet')

def animate(i):

    global pdf
    omega = 1
    pdf_streamed = streaming(pdf)

    assert np.allclose(np.sum(pdf), np.sum(pdf_streamed), atol=1e-3) # check mass conservation
    
    # recalculate the density (rho)
    density = calc_density(pdf_streamed)

    #print("density = ", density)
    #print("density shape = ", density.shape)
    #print("density average = ", np.average(density))

    # calculate local average velocity (u)
    local_avg_velocity = calc_local_avg_velocity(pdf_streamed)
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
    
    # collision step using np.roll
    pdf_collision = pdf_streamed + omega*(equilibrium_pdf - pdf_streamed)
    """
    print("pdf_collision = ", pdf_collision)
    print("pdf_collision shape = ", pdf_collision.shape)
    """


    assert np.allclose(np.sum(pdf_streamed), np.sum(pdf_collision), atol=1e-3 )# check mass conservation

    pdf = pdf_collision
    # update the image
    im.set_array(calc_density(pdf))
    return im,

anim = animation.FuncAnimation(fig, animate, frames=300, interval=200, blit=True)
#HTML(anim.to_html5_video())
anim.save("results/ml2_anim.gif", writer = 'pillow', fps = 30)