import sys
import numpy as np
from random_generators import *

N = 50000000
t = float(sys.argv[1])
mcc = mcCherenkov(t,N)

fname = 'gg_delta_theta_t%.0f.npy'%t if t >=0 else 'gg_delta_theta_tm%.0f.npy'%np.abs(t)
ratio_fname = 'r_delta_t%.0f.npy'%t if t >=0 else 'r_delta_tm%.0f.npy'%np.abs(t)
np.save(fname,mcc.gg_array)
np.save(ratio_fname,mcc.ratio_array)
