import argparse
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
    parser.add_argument('--omega', type=float, default=0.1,
                        help='Relaxation parameter')
    parser.add_argument('--mode', type=str, default="shear_wave_2",
                        help='Mode of simulation')
    parser.add_argument('--parallel', action='store_true', # default is False
                        help='Parallel simulation')
    args = parser.parse_args()

    lbm = LBM(NX=args.NX, NY=args.NY, parallel=args.parallel, mode = args.mode)
    lbm.run(nt=args.nt)

    p = "_parallel" if args.parallel else "_serial"

    lbm.plot_density(filename="density_"+args.mode+p+".gif")
    lbm.plot_velocity(filename="velocity_"+args.mode+p+".gif")

