from random_generators import *
import matplotlib.pyplot as plt

lEs = np.linspace(1, np.log(1.e11), 10000)

qcdfs = np.empty((lEs.size,2,10000))

for i,lE in enumerate(lEs):
    qg = Qgen(lE)
    cdf = np.empty((2,10000))
    cdf[0] = qg.qs
    cdf[1] = qg.cdf
    qcdfs[i] = cdf

# plt.ion()
# plt.figure()
# for i in range(lEs.size):
#     plt.plot(qcdfs[i,0],qcdfs[i,1])

np.savez('qecdf_lE.npz',qecdf_lE=qcdfs,lEs=lEs)
