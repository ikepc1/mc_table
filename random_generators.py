import numpy as np
from cherenkov_photon import CherenkovPhoton as cp
from charged_particle import EnergyDistribution, AngularDistribution
import scipy.stats as st
from scipy.constants import value,nano
from scipy.integrate import quad, cumtrapz
from cherenkov_photon_array import CherenkovPhotonArray as cpa


class Egen(EnergyDistribution):
    """This is a class for drawing random charged particle energies"""
    def __init__(self, t, ul):
        super().__init__('Tot',t, ul)
        self.t = t
        self.lEs = np.linspace(self.ll,self.ul,1000000)
        self.cdf = self.make_cdf(self.lEs)

    def make_cdf(self,lEs):
        cdf = np.empty_like(lEs)
        cdf[0] = 0.
        cdf[1:] = cumtrapz(self.spectrum(lEs),lEs)
        cdf /= cdf.max()
        return cdf

    def gen_lE(self, N=1):
        rvs = st.uniform.rvs(size=N)
        return np.interp(rvs,self.cdf,self.lEs)

class Qgen(AngularDistribution):
    """This is a class for drawing random charged particle angles"""

    def __init__(self, lE):
        super().__init__(lE)
        # self.qs = np.linspace(self.lls[0],self.uls[-1],10000)
        self.ul = self.set_upper_lim(lE)
        self.qs = np.logspace(-9,np.log10(self.ul),10000)
        self.cdf = self.make_cdf(self.qs)

    def set_upper_lim(self,lE):
        if self.log10E > 3.:
            return np.pi - (np.pi / np.log(1.e12)) * lE
        else:
            return np.pi

    def make_cdf(self,qs):
        cdf = np.empty_like(qs)
        cdf[0] = 0.
        cdf[1:] = cumtrapz(self.norm_integrand(qs),qs)
        # print(cdf.max())
        cdf /= cdf.max()
        return cdf

    def gen_theta(self,N=1):
        rvs = st.uniform.rvs(size=N)
        return np.interp(rvs,self.cdf,self.qs), rvs

class Qgen_from_table():
    """This is a class for drawing random charged particle angles"""
    q_table_file = 'qecdf_lE.npz'
    def __init__(self):
        self.table = np.load(self.q_table_file)
        self.lEs = self.table['lEs']
        self.cdfs = self.table['qecdf_lE']

    def gen_theta(self,lE,N=1):
        rvs = st.uniform.rvs(size=N)
        diff = np.abs(lE-self.lEs)
        # i_lE = np.searchsorted(self.lEs,lE)
        i_lE = np.abs(lE-self.lEs).argmin()
        cdf = self.cdfs[i_lE]
        return np.interp(rvs,cdf[1],cdf[0]), rvs


class mcCherenkov():
    """docstring for ."""
    c = value('speed of light in vacuum')
    hc = value('Planck constant in eV s') * c
    table_file = 'gg_t_delta_theta_2020_normalized.npz'
    ul = np.log(1.e11) #energy upper limit
    table = cpa(table_file)
    min_lE = np.log(cp.cherenkov_threshold(table.delta.max()))

    def __init__(self, t, Nch, min_l = 300, max_l = 600):
        self.t = t
        self.Egen = Egen(self.t, self.ul)
        self.Qgen = Qgen_from_table()
        lE_array = self.throw_lE(Nch)
        self.lE_array = lE_array[lE_array>self.min_lE]
        self.theta_e = self.make_theta_e(self.lE_array)
        self.theta_bins, self.mid_theta_bins = self.make_bins()
        # self.gg_list = self.make_gg_list()
        self.gg_array = self.make_gg_array()

    def make_gg_t_delta(self,delta):
        lE_Cher_bool = self.throw_gamma(self.lE_array,delta)
        lE_Cher = self.lE_array[lE_Cher_bool]
        theta_e = self.theta_e[lE_Cher_bool]
        theta_g = cp.cherenkov_angle(np.exp(lE_Cher),delta)
        phi = self.throw_phi(lE_Cher.size)
        theta = cp.spherical_cosines(theta_e,theta_g,phi)
        return self.make_gg(theta)

    def make_gg_list(self):
        gg_list = []
        for i,d in enumerate(self.table.delta):
            gg_list.append(self.make_gg_t_delta(d))
        return gg_list

    def make_gg_array(self):
        gg_array = np.empty((self.table.delta.size,self.table.theta.size))
        for i,d in enumerate(self.table.delta):
            gg_array[i] = self.make_gg_t_delta(d)
        return gg_array

    def throw_lE(self, N=1):
        '''
        Draw values from normalized energy distribution for stage t

        parameters:
        t : stage to set energy distribution
        N : number of lEs to be drawn

        returns:
        array of log energies (MeV) of size N
        '''
        return self.Egen.gen_lE(N)

    def throw_qe(self, lE, N=1):
        '''
        Draw values from normalized angular distribution for particles of
        log energy lE

        parameters:
        lE : log energy (MeV) to set energy distribution
        N : number of thatas to be drawn

        returns:
        array of thetas (radians) of size N
        '''
        return Qgen(lE).gen_theta(N)[0]

    def throw_qe_table(self, lE, N=1):
        '''
        Draw values from normalized angular distribution for particles of
        log energy lE

        parameters:
        lE : log energy (MeV) to set energy distribution
        N : number of thatas to be drawn

        returns:
        array of thetas (radians) of size N
        '''
        return self.Qgen.gen_theta(lE,N)[0]

    def throw_phi(self,N=1):
        return 2*np.pi*st.uniform.rvs(size=N)

    def cherenkov_dE(self,min_l,max_l):
        return self.hc/(min_l*nano) - self.hc/(max_l*nano)

    def max_yield(self,delta,min_l,max_l):
        '''
        This function returns the max possible Cherenkov yield of a hyper
        relativistic charged parrticle.

        Parameters:
        delta: atmospheric delta at which to calculate the yield
        min_l: minimum cherenkov wavelength
        max_l: maximum cherenkov wavelength

        returns:
        the number of cherenkov photons per meter per charged particle
        '''
        alpha_over_hbarc = 370.e2
        chq = cp.cherenkov_angle(1.e12,delta)
        return alpha_over_hbarc*np.sin(chq)**2*self.cherenkov_dE(min_l,max_l)

    def throw_gamma(self,lEs,delta):
        cy = cp.cherenkov_yield(np.exp(lEs), delta)
        return st.uniform.rvs(size=lEs.size) < cy

    def make_theta_e(self,lEs):
        '''
        Make an array of drawn theta_e's corresponding to the array of log
        energies lEs
        '''
        theta_e = np.empty_like(lEs)
        for i,lE in enumerate(lEs):
            # theta_e[i] = self.throw_qe(lE)
            theta_e[i] = self.throw_qe_table(lE)
        return theta_e

    def calculate_theta(self,lEs):
        '''
        Make an array of Cherenkov photon angles corresponding to an array of
        Cherenkov producing log energies (lEs)
        returns:
        theta: array of Cherenkov photon angles (with respect to the shower axis)
        theta_e: array of charged particle angles
        theta_g: array of Cherenkov photon angles (with respect to the charged
        particle travel direction)
        phi: array of cherenkov photon azimuthal angles (with respect to the charged
        particle travel direction)
        '''
        theta_e = self.make_theta_e(lEs)
        theta_g = cp.cherenkov_angle(np.exp(lEs),self.delta)
        phi = self.throw_phi(lEs.size)
        return cp.spherical_cosines(theta_e,theta_g,phi), theta_e, theta_g, phi

    def make_ecdf(self,theta):
        sorted_q = np.sort(theta)
        return (np.arange(theta.size) + 1)/theta.size, sorted_q

    def make_bins(self):
        half_diff = np.diff(np.log(self.table.theta))[0]/2
        lgtheta_bins = np.log(self.table.theta) - half_diff
        lgtheta_bins = np.append(lgtheta_bins, lgtheta_bins[-1] + half_diff)
        theta_bins = np.exp(lgtheta_bins)
        mid_theta_bins = theta_bins[:-1] + np.diff(theta_bins) / 2.
        return theta_bins, mid_theta_bins

    def make_gg(self,theta):
        h,b = np.histogram(theta,bins=self.theta_bins,weights = 1/np.sin(theta),density=True)
        int_midpoint = np.sum(h*np.sin(self.mid_theta_bins)*4*np.pi*np.diff(self.theta_bins))
        return h / int_midpoint




class table_CDF(cpa):
    def __init__(self, table, t, delta):
        super().__init__(table)
        self.cdf = self.make_cdf(t, delta, self.theta)

    def cdf_integrand(self, t, delta, theta):
        gg = self.interpolate_gg(t,delta,theta)
        return gg * np.sin(theta) * 4 * np.pi

    def make_cdf(self, t, delta, theta):
        cdf = np.empty_like(theta)
        cdf[0] = 0.
        cdf[1:] = cumtrapz(self.cdf_integrand(t, delta, theta),theta)
        cdf /= cdf.max()
        return cdf

    def cdf_function(self,theta):
        return np.interp(theta,self.theta,self.cdf)

    def interpolate_gg(self, t, delta, theta):
        '''This funtion returns the interpolated values of gg at a given delta
        and theta
        parameters:
        t: single value of the stage
        delta: single value of the delta
        theta: array of theta values at which we want to return the angular
        distribution

        returns:
        the angular distribution values at the desired thetas
        '''

        gg_td = self.angular_distribution(t,delta)
        return np.interp(theta,self.theta,gg_td)

    def gen_theta(self,N=1):
        rvs = st.uniform.rvs(size=N)
        return np.interp(rvs,self.cdf,self.theta), rvs




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cherenkov_photon_array import CherenkovPhotonArray as cpa
    import time
    plt.ion()

    t = -6.
    N = 50000

    start_time = time.time()
    mcc = mcCherenkov(t,N)
    end_time = time.time()
    print("Generated cmc gg list for one stage in %.1f s"%(
        end_time-start_time))

    plt.figure()
    delta = mcc.table.delta[50]
    plt.plot(mcc.table.theta,mcc.table.angular_distribution(t,delta), label = 'table')
    plt.plot(mcc.mid_theta_bins,mcc.gg_array[50], label = 'mc')
    plt.loglog()
    plt.legend()

    # table_file = 'gg_t_delta_theta_2020_normalized.npz'
    #
    # #plot theta histogram and table pdf
    # plt.figure()
    # table = cpa(table_file)
    # min_Omega = -2 * np.pi * (np.cos(mcc.theta.min()) - 1)
    # d_omega_bins = np.linspace(min_Omega,np.pi,100000)
    # d_theta_bins = np.arccos(1 - d_omega_bins / (2*np.pi))
    # h,b = np.histogram(mcc.theta,bins = d_theta_bins)
    # mid_theta_bins = d_theta_bins[:-1] + np.diff(d_theta_bins) / 2.
    # int_midpoint = np.sum(h*np.sin(mid_theta_bins)*4*np.pi*np.diff(d_theta_bins))
    # int_trapz = np.trapz(h*np.sin(d_theta_bins[:-1])*4*np.pi,d_theta_bins[:-1])
    # h_mid = h / int_midpoint
    # h_trapz = h / int_trapz
    # plt.hist(d_theta_bins[:-1],bins = d_theta_bins, weights = h_mid, histtype = 'step', label = 'thrown')
    # plt.loglog()
    # plt.plot(mcc.mid_bins,mcc.gg)
    # plt.plot(table.theta,table.angular_distribution(t,delta), label = 'table (for reference)')
    # plt.legend()
    # plt.title('%d MC trial Cherenkov distribution for stage = %.0f, and delta = %.4f'%(N,t,delta))
    # plt.xlabel('theta (rad)')
    # plt.ylabel('dN_gamma / dOmega')
    #
    # #plot ecdf and cdf comparison
    # plt.figure()
    # plt.plot(np.sort(mcc.theta),mcc.ecdf, label = 'MC ecdf')
    #
    # tcdf = table_CDF(table_file,t,delta)
    # table_sample = tcdf.gen_theta(N)[0]
    # table_sample_ecdf = (np.arange(N) + 1) / N
    # ks, p  = st.kstest(mcc.theta,tcdf.cdf_function)
    #
    # plt.plot(tcdf.theta,tcdf.cdf, label = 'table cdf')
    # plt.plot(np.sort(table_sample),table_sample_ecdf, label = 'table sample ecdf')
    # plt.legend()
    # plt.xlabel('theta (rad)')
    # plt.ylabel('cdf')
    # plt.title('ks stat = %.3f, p value = %f'%(ks,p,))
    # plt.semilogx()
    #
    #
    # #plot energy histograms
    # plt.figure()
    # h,bins = np.histogram(np.exp(mcc.lE_array),bins = 100)
    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    # plt.hist(np.exp(mcc.lE_array),bins = logbins,histtype = 'step',label='all energies')
    # plt.hist(np.exp(mcc.lE_above),bins = logbins,histtype = 'step',label='above threshold')
    # plt.hist(np.exp(mcc.lE_Cher),bins = logbins,histtype = 'step',label='Cherenkov producing')
    # plt.semilogx()
    # plt.title('Charged Particle MC Energy histogram for t = %.0f'%t)
    # plt.legend()
