import sys
from random_generators import *
import numpy as np
from cherenkov_photon_array import CherenkovPhotonArray as cpa

t = float(sys.argv[1])

table_file = 'gg_t_delta_theta_2020_normalized.npz'
table = cpa(table_file)
N = 1000000

lgtheta_bins = np.log(table.theta) - np.diff(np.log(table.theta))[0]/2
theta_bins = np.exp(lgtheta_bins)

for delta in table.delta:
    mcc = mcCherenkov(t,delta,N,np.log(1.e11))
    np.savez_compressed('N1e6_t%.0f_delta%.5f'%(t,delta),theta=mcc.theta,t=mcc.t,delta=mcc.delta)
