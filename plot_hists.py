import numpy as np
import matplotlib.pyplot as plt
from cherenkov_photon_array import CherenkovPhotonArray as cpa
from scipy.signal import savgol_filter
plt.ion()

table_file = 'gg_t_delta_theta_2020_normalized.npz'
mcc = np.load('CMCN1e7_t-6_d1e-4.npz')
table = cpa(table_file)

#create bins with equal omega
min_Omega = 2 * np.pi * (1 - np.cos(mcc['theta'].min()))
d_omega_bins = np.linspace(min_Omega,np.pi,50000)
d_theta_bins = np.arccos(1 - d_omega_bins / (2*np.pi))
h,b = np.histogram(mcc['theta'],bins = d_theta_bins)
mid_theta_bins = d_theta_bins[:-1] + np.diff(d_theta_bins) / 2.
int_midpoint = np.sum(h*np.sin(mid_theta_bins)*4*np.pi*np.diff(d_theta_bins))
int_trapz = np.trapz(h*np.sin(d_theta_bins[:-1])*4*np.pi,d_theta_bins[:-1])
h_mid = h / int_midpoint
h_trapz = h / int_trapz



plt.figure()
plt.hist(d_theta_bins[:-1],bins = d_theta_bins , weights = h_mid, histtype = 'step', label = 'thrown, normalized with midpoint Riemann')
plt.hist(d_theta_bins[:-1],bins = d_theta_bins, weights = h_trapz, histtype = 'step', label = 'thrown, normalized with trapezoidal sum')
plt.plot(table.theta,table.angular_distribution(mcc['t'],mcc['delta']), label = 'table (for reference)')
plt.loglog()
plt.legend()
plt.xlabel('theta (rad)')
plt.ylabel('dN_gamma / dOmega')

#bin in theta using Freedman-Diaconis, then weight each bin using # of steradians in each bin
plt.figure()
h,fd_theta_bins = np.histogram(mcc['theta'], bins = 'fd')
cos = np.cos(fd_theta_bins)
omega_in_bin = 2 * np.pi * (cos[:-1] - cos[1:])
plt.hist(fd_theta_bins[:-1], bins = fd_theta_bins, weights = h / omega_in_bin, histtype = 'step', density = True)
plt.plot(table.theta,table.angular_distribution(mcc['t'],mcc['delta']), label = 'table (for reference)')
plt.loglog()
plt.xlabel('theta (rad)')
plt.ylabel('dN_gamma / dOmega')

#bin directly and weight each contribution based on it's contribution to density

lgtheta_bins = np.log(table.theta) - np.diff(np.log(table.theta))[0]/2
theta_bins = np.exp(lgtheta_bins)
plt.figure()
h,b = np.histogram(mcc['theta'],bins=theta_bins,weights = 1/np.sin(mcc['theta']),density=True)
mid_theta_bins = theta_bins[:-1] + np.diff(theta_bins) / 2.
int_midpoint = np.sum(h*np.sin(mid_theta_bins)*4*np.pi*np.diff(theta_bins))
h_mid = h / int_midpoint
h_mid_smoothed = h_mid
h_mid_smoothed[:100] = savgol_filter(h_mid[:100],51,10)
plt.hist(theta_bins[:-1],bins = theta_bins , weights = h_mid, histtype = 'step',label='original')
plt.hist(theta_bins[:-1],bins = theta_bins , weights = h_mid_smoothed, histtype = 'step',label='smoothed')
plt.plot(table.theta,table.angular_distribution(mcc['t'],mcc['delta']), label = 'table (for reference)')
plt.plot(mid_theta_bins,h_mid)
plt.plot(mid_theta_bins,h_mid_smoothed)

plt.legend()
# plt.hist(mcc['theta'],bins=theta_bins,weights=1/(4*np.pi*np.sin(mcc['theta'])),density=True,histtype='step')
plt.loglog()
plt.xlabel('theta (rad)')
plt.ylabel('dN_gamma / dOmega')
