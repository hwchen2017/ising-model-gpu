import numpy as np
import math
import matplotlib 
from matplotlib import pyplot


temp, ene, enesq, mag, magsq = [], [], [], [], []

filename = "energy_magnetization_1024x1024_200000.txt"

for line in open(filename):
    tmp = line.strip().split(' ')
    tmp = list(map(float, tmp))
    temp.append(tmp[0])
    ene.append(tmp[1])
    enesq.append(tmp[2])
    mag.append(tmp[3])
    magsq.append(tmp[4])

mag = np.array(mag)
magsq = np.array(magsq)
ene = np.array(ene)
enesq = np.array(enesq)

# pyplot.plot(temp, mag)
# pyplot.plot(temp, mag, 'o')
# pyplot.ylabel("Magnetization")
# pyplot.xlabel("Temperature")

pyplot.plot(temp, magsq - mag**2)
pyplot.plot(temp, magsq - mag**2, 'o')
pyplot.ylabel("Susceptibility")
pyplot.xlabel("Temperature")

pyplot.show()