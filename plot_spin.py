import numpy as np
import matplotlib.pyplot as pyplot

lattice = np.loadtxt("spin_config_1024x1024_100_temp_2..txt")
pyplot.imshow(lattice)
pyplot.title('Ground State Lattice Configuration')
pyplot.colorbar()
pyplot.show()