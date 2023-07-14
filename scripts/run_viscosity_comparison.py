import argparse
import numpy as np
import matplotlib.pyplot as plt
from hpc_fluid_dynamics.lbm_class import LBM

"""
to run the script using mpi :
mpirun -np 4 python scripts/run_lbm.py
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LBM')
    parser.add_argument('--NX', type=int, default=250,
                        help='Number of lattice points in the x direction')
    parser.add_argument('--NY', type=int, default=250,
                        help='Number of lattice points in the y direction')
    parser.add_argument('--nt', type=int, default=1000,
                        help='Number of time steps')
    parser.add_argument('--parallel', action='store_true', # default is False
                        help='Parallel simulation')
    args = parser.parse_args()
    

    omegas = np.arange(0.1, 1.9, 0.1)
    v = []
    m_v = []

    for omega in omegas:
        lbm = LBM(NX=args.NX, NY=args.NY, parallel=args.parallel, mode = "shear_wave", omega=omega)
        lbm.run(nt=1000)
        v.append(lbm.viscosity)
        m_v.append(lbm.measured_viscosity)

    plt.plot(omegas, v, label="theoretical viscosity")
    plt.plot(omegas, m_v, label="measured viscosity")
    plt.xlabel("omega")
    plt.ylabel("viscosity")
    plt.legend()
    plt.savefig("results/viscosity_shear_wave.png")
