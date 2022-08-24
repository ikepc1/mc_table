import numpy as np

gg_t_delta_theta = np.zeros((21,176,321),dtype=float)
r_delta_t = np.zeros((21,176),dtype=float)
for i in range(21):
  t = 2*i-20
  if t>=0:
    fname = 'gg_delta_theta_t%d.npy'%t
    ratio_fname = 'r_delta_t%d.npy'%t
  else:
    fname = 'gg_delta_theta_tm%d.npy'%abs(t)
    ratio_fname = 'r_delta_tm%d.npy'%abs(t)
  gg_t_delta_theta[i] = np.load(fname)
  r_delta_t[i] = np.load(ratio_fname)
t = np.linspace(-20,20,21,dtype=float)
delta = np.logspace(-7,-3.5,176)
theta = np.logspace(-3,0.2,321)
np.savez('gg_t_delta_theta_mc.npz',gg_t_delta_theta=gg_t_delta_theta,t=t,delta=delta,theta=theta)
np.savez('r_t_delta.npz',r_delta_t = r_delta_t, t=t, delta=delta)
