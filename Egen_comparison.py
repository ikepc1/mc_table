from random_generators import *
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
N = 1000000
ul_array = np.linspace(np.log(1.e6),np.log(1.e12),100)
lE_array = np.empty((ul_array.size,N))
for i,ul in enumerate(ul_array):
    eg = Egen(-6.,ul)
    lE_array[i,:] = eg.gen_lE(N)[0]
plt.figure()
plt.plot(np.exp(ul_array),np.exp(lE_array.max(axis=1)))
plt.xlabel('upper limit for E norm integral (MeV)')
plt.ylabel('maximum energy of N = %d trials (MeV)'%N)
