from random_generators import *
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

lEs = np.linspace(np.log(1.e0),np.log(1.e10),10)
plt.figure()
for lE in lEs:
    qg = Qgen(lE)
    plt.plot(qg.qs,qg.n_t_lE_Omega(qg.qs),label = '%.1f'%np.exp(lE))

plt.semilogx()
plt.legend()
