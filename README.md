# High performance computing - Fluid dynamics simulation

This repository contains a fluid dynamics simulation project implemented using high performance computing techniques in Python.

## Overview 

The code implements a Lattice Boltzmann Method (LBM) fluid simulation using MPI for parallelization. Key aspects include:

-   LBM implementation for fluid flow modeling
-   Parallel processing using MPI domain decomposition
-   Simulation of different test cases like lid-driven cavity, Poiseuille flow
-   Visualization of simulation results


## Installation

The code uses the following packages:

    mpi4py
    numpy
    matplotlib

To install the packages, run the following command:
    
    ```bash 
    conda create --name hpc_env
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage
### Running the simulation (serial)
To run the LBM simulation in serial, run the following command:

    ```bash
    python3 scripts/run_lbm.py --mode lid --NX 300 --NY 300  --nt 1000
    ```

The following arguments are available:

    --mode: Simulation mode. Possible values: lid, poiseuille, couette, shear_wave
    --omega: Relaxation parameter
    --NX: Number of grid points in X direction
    --NY: Number of grid points in Y direction
    --nt: Number of time steps
    --no_plot: add this flag to disable plotting. Otherwise, animations will be outputted to the results/ folder

### Running the simulation (parallel)
To run the LBM simulation in parallel, add a `--parallel` flag to the above command. 

### Time benchmarking
To plot simulation time against number of processes, run the following command:

    ```bash
    python3 scripts/time_measure.py
    ```

### Running on the cluster
To run the simulation on the cluster, run the following command:

    ```bash
    batch scripts/cluster/lbm.job
    ```
