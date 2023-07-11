from hpc_fluid_dynamics.lbm_class import LBM

"""
to run the script using mpi :
mpirun -np 4 python scripts/run_lbm.py
"""

lbm = LBM(NX=50, NY=50, parallel=True)
lbm.run(nt=100)
lbm.plot_density(filename="density_parallel.gif")
lbm.plot_velocity(filename="velocity_parallel.gif")