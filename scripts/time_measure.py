"""
Measure execution time for parallel simulation
"""

import time
import os
import matplotlib.pyplot as plt

procs_nbs = [1, 4, 9, 16, 25, 36, 64, 100]
procs_times = []
for procs_nb in procs_nbs:
    start = time.time()
    cmd = "mpirun -np {} python scripts/run_lbm.py --NX 600 --NY 600 --parallel --no_plot --nt 300".format(procs_nb)
    os.system(cmd)
    end = time.time()
    print("Execution time for {} processes: {} seconds".format(procs_nb, end-start))
    procs_times.append(end-start)


plt.plot(procs_nbs, procs_times)
plt.xlabel("Number of processes")
plt.ylabel("Execution time (s)")
plt.title("Execution time for parallel simulation")
plt.savefig("results/execution_time_parallel.png")
plt.show()