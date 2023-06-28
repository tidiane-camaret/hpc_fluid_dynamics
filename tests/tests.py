import numpy as np
import matplotlib.pyplot as plt

L, W = 50, 30 

fig = plt.figure()
ax = plt.axes()
im = ax.imshow(np.ones((L, W)) , cmap='jet')
plt.show()