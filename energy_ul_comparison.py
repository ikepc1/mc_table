from random_generators import *
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

delta = 1.e-4
t = -6.
N = 100000

table_file = 'gg_t_delta_theta_2020_normalized.npz'
tcdf = table_CDF(table_file,t,delta)

ul_array = np.linspace(np.log(1.e6),np.log(1.e10),10)
e_array = np.exp(ul_array)
ks = np.empty_like(ul_array)
p = np.empty_like(ul_array)
plt.figure()
plt.plot(tcdf.theta,tcdf.cdf, label = 'table cdf')
mcclist = []
for i,ul in enumerate(ul_array):
    print(ul)
    mcc = mcCherenkov(t,delta,N,ul)
    mcclist.append(mcc)
    plt.plot(np.sort(mcc.theta),mcc.ecdf, label = 'MC ecdf ul =  %.2E'%ul)
    ks[i], p[i]  = st.kstest(mcc.theta,tcdf.cdf_function)

    plt.legend()
plt.semilogx()

for i,mcc in enumerate(mcclist):
    plt.figure()
    h,bins = np.histogram(np.exp(mcc.lE_array),bins = 100)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(np.exp(mcc.lE_array),bins = logbins,histtype = 'step',label='all energies')
    plt.hist(np.exp(mcc.lE_above),bins = logbins,histtype = 'step',label='above threshold')
    plt.hist(np.exp(mcc.lE_Cher),bins = logbins,histtype = 'step',label='Cherenkov producing')
    plt.semilogx()
    plt.title('Upper limit = %.3E MeV'%np.exp(ul_array[i]))
    # plt.title('Charged Particle MC Energy histogram for t = %.0f'%t)
    plt.legend()

plt.figure()
plt.plot(e_array,ks)
plt.semilogx()
plt.title('KS Statistic vs Upper bound on energy normalization')

plt.figure()
plt.plot(e_array,p)
plt.semilogx()
plt.title('KS P-value vs Upper bound on energy normalization')
