### MAGNETAR BURST PHASE DEPENDENCE ###

#%% set working directory

import os

work_dir = "/Users/DexterIII/Documents/Werk/PhD/research/projects/burst_phase/work_folder"

os.chdir(work_dir)

#%% import relevant packages

import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time as tm
import sys

from numpy import pi,radians,arcsin,cos,sin,sqrt,arccos,exp,log
from math import ceil
from scipy import integrate, interpolate
from scipy.stats import poisson

#%% define A(r), i.e. Schwarzschild correction factor
    
def A(r):
    return 1 - 1 / r

#%% theta_star(b) for  0 <= b < b_max

if not os.path.isdir("theta_star_inf_out"):
       os.makedirs("theta_star_inf_out")

def theta_star_inf(R = 2.5,N_u = 1e5,N_b = 1e5):
    
    b_max = R * A(R)**(-1/2)
    
    def integral(b):
        
        intgrand = lambda u: b * (1 - (1 - u) * (u * b)**2)**(-1/2)
    
        u = np.linspace(0,1 / R,int(N_u),endpoint=True)
        theta_star = [intgrand(u_i) for u_i in u]
        return integrate.simps(theta_star,u)
    
    b_array = np.linspace(0,b_max * (1 - 1e-6),int(N_b),endpoint=True)
    theta_star_array = integral(b_array)
    theta_star_data = np.transpose([b_array,theta_star_array])
    
    logN_u = int(np.log10(N_u))
    logN_b = int(np.log10(N_b))
    
    np.savetxt("theta_star_inf_out/theta_star_inf_R_{0}_Nu_{1}_Nb_{2}".format(R,logN_u,logN_b),theta_star_data,delimiter=" ")

#%%

#theta_star_inf()

#%%

R = 2.5
N_u = 1e5
N_b = 1e5

b_max = R * A(R)**(-1/2)

b, theta_star = np.transpose(np.loadtxt("theta_star_inf_out/theta_star_inf_R_{0}_Nu_{1}_Nb_{2}".format(R,int(np.log10(N_u)),int(np.log10(N_b)))))
theta_star_b = interpolate.interp1d(b,theta_star)

cSB = "SteelBlue"

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(b,theta_star,color = cSB) 
ax.axvline(b_max,color='k',linestyle="dashed")
ax.set_xlabel(r"$b$")
ax.set_ylabel(r"$ \theta_*(b) $")
ax.set_title(r"$R = {0}$".format(R))
plt.show()

#%%

if not os.path.isdir("LBS_out"):
       os.makedirs("LBS_out")

def generate_LBS(R = 2.5,psi = 1,beam = 0):

    N_b = 1e5
    rpsi = radians(psi)
    
    b_max = R * A(R)**(-1/2)

    if beam == 1:
        sigma = pi / 6
    else:
        sigma = 1

    # beaming function: beam = {0:isotropic,1:anisotropic -> gaussian} 
    def f(b):
        if beam == 0:
            return 1
        if beam == 1:
            def delta(b):
                return arcsin(b / b_max)
            return sqrt(pi) / (sp.special.erf(pi / (2 * sqrt(2) * sigma))) * (2 * sigma**2)**(-1/2) * exp(- delta(b)**2 / (2 * sigma**2))
       
    def Phi(b,theta_0):
        if (theta_0 - rpsi) < theta_star_b(b) < (theta_0 + rpsi):
            h1 = cos(rpsi) - cos(theta_0) * cos(theta_star_b(b))
            h2 = sin(theta_0) * sin(theta_star_b(b))
            return 2 * arccos(h1 / h2)
        else:
            return 0
    
    def Phi_zero(b):
        if 0 <= theta_star_b(b) < rpsi:
            return 2 * pi 
        else:
            return 0

    def kappa(theta_0):
    
        kappa1 = (R**(1/2) / b_max)**4
        b_array = np.linspace(0,b_max * (1-1e-6),N_b,endpoint=True)
    
        if theta_0 == 0:
            intgrnd_kappa2 = lambda b : Phi_zero(b) * b * f(b)
        else: 
            intgrnd_kappa2 = lambda b : Phi(b,theta_0) * b * f(b)
        
        kappa2 = [intgrnd_kappa2(b_i) for b_i in b_array]
        kappa2_b = integrate.simps(kappa2,b_array)
    
        return kappa1 * kappa2_b

    N_theta_0 = 360
    theta_0_array = radians(np.linspace(0,180,N_theta_0 + 1))

    I_obs = [kappa(theta_0_i) for theta_0_i in theta_0_array]

    indx_nan = np.where(np.isnan(I_obs))[0]
    I_obs = np.delete(I_obs,indx_nan)
    theta_0_array = np.delete(theta_0_array,indx_nan)

    np.savetxt("LBS_out/LBS_R_{0}_psi_{1}_beam_{2}_test".format(R,psi,beam),np.transpose([theta_0_array,I_obs]))

#%%

#generate_LBS()

#%%

R = 2.5
psi = 1
beam = 0

theta, LBS = np.transpose(np.loadtxt("LBS_out/LBS_R_{0}_psi_{1}_beam_{2}".format(R,psi,beam)))
LBS_theta = interpolate.interp1d(theta,LBS)

indx_terminator = np.where(LBS <= 1e-7)[0][0]

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(theta / pi,LBS_theta(theta),color=cSB) 
ax.axvline(theta[indx_terminator] / pi,color='k',linestyle="dashed")
ax.set_xlabel(r"$\theta_0 / \pi$")
ax.set_ylabel(r"$ \kappa(\theta_0) $")
ax.set_title(r"$R = {0}$".format(R))
plt.show()

#%%

def norm_LBS(phi,R,psi,beam,alpha,chi):
    
    theta, LBS = np.transpose(np.loadtxt("LBS_out/LBS_R_{0}_psi_{1}_beam_{2}".format(R,psi,beam)))
    LBS_theta = interpolate.interp1d(theta,LBS)
    
    alpha_rad = radians(alpha)
    chi_rad = radians(chi)

    def theta_0(phi):
        return arccos(cos(alpha_rad) * cos(chi_rad) + sin(alpha_rad) * sin(chi_rad) * cos(phi))
    
    phi_bins = int(1e5)
    phi_array = np.linspace(0,2 * pi,phi_bins + 1,endpoint=True)
    A = integrate.simps(LBS_theta(theta_0(phi_array)),phi_array)

    f_norm = interpolate.interp1d(phi_array,LBS_theta(theta_0(phi_array)) / A)

    return f_norm(phi)

#%%

alpha = 90
chi = 90

phi_lst = np.linspace(0,2 * pi,int(1e3))

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(phi_lst / (2 * pi),norm_LBS(phi_lst,R,psi,beam,alpha,chi),color=cSB) 
ax.axvline(theta[indx_terminator] / (2 * pi),color='k',linestyle="dashed")
ax.axvline(1 - theta[indx_terminator + 1] / (2 * pi),color='k',linestyle="dashed")
ax.set_xlim(0,1)
ax.set_xlabel(r"$\phi/2\pi$")
ax.set_ylabel(r"$ \rm{Normalized\ LBS} $")
ax.set_title(r"$R = {0}$".format(R))
plt.show()

#%%

## Skewed (or Asymmetric) Laplace distribution
#   pdf :   probability density function
#   cdf :   cumulative density function
#   ppf :   percent point function (inverse of the cdf)

# t0: peak time
# A: scale parameter
# s: skew parameter  

# skewed Laplace probability density function
def skewed_Laplace_pdf(t,t0,A,s):
    t_t0 = lambda t: A / (s + 1 / s) * exp(A * s * (t - t0))
    t0_t = lambda t: A / (s + 1 / s) * exp(- A * (t - t0) / s)
    return np.piecewise(t, [t < t0, t >= t0], [t_t0,t0_t]) 

# skewed Laplace cumulative density function
def skewed_Laplace_cdf(t,t0,A,s):    
    t_t0 = lambda t: 1 / (1 + s**2) * exp(A * s * (t - t0))
    t0_t = lambda t: 1 - s**2 / (1 + s**2) * exp(- A * (t - t0) / s)
    return np.piecewise(t, [t <= t0, t > t0], [t_t0,t0_t]) 

# skewed Laplace point percent function
def skewed_Laplace_ppf(q,t0,A,s):
    XminM = lambda q: t0 + log((1 + s**2) * q) / (A * s)
    XplusM = lambda q: t0 - log((1 - q) * (1 + s**2) / s**2) * s / A
    return np.piecewise(q, [t0 > XminM(q), t0 < XplusM(q)], [XminM(q),XplusM(q)])
        
#%%

# the model is the skewed Laplace probability density function with normalisation *N* and background rate *bg_rate*
def burst_model_pdf(t,t0,A,s,N,bg_rate):
    return N * skewed_Laplace_pdf(t,t0,A,s) + bg_rate

def burst_model_cdf(t,t0,A,s,N,bg_rate,offset):
    return N * skewed_Laplace_cdf(t,t0,A,s) + bg_rate * t + offset

#def burst_model_ppf(q,t0,A,s,N,bg_rate):
#    XminM = lambda q: q / bg_rate - (s * sp.special.lambertw((A * np.exp((A * (q - bg_rate * t0)) / (bg_rate * s)) * N * s) / (bg_rate * (1 + s**2)))) / A
#    XplusM = lambda q: (- N + q) / bg_rate + sp.special.lambertw((A * np.exp((A * s * (N - q + bg_rate * t0)) / bg_rate) * N * s) / (bg_rate * (1 + s**2))) / (A * s)
#    return np.piecewise(q, [t0 > XminM(q), t0 < XplusM(q)], [XminM(q),XplusM(q)])

def DT_burst(ysig,A,s,N,b,bw):
    return - (s + 1 / s)/ A * log((s + 1 / s)/(N * A * bw) * (ysig - b))

def C1(s,Dt):
    return (s + 1 / s) / Dt
    
def C2(ysig,s,b,bw):
    return (s + 1 / s) * (ysig - b) / bw

def Ncounts(ysig,A,s,b,bw,Dt):
    return C2(ysig,s,b,bw) * exp(A / C1(s,Dt)) / A
    
#%%

logmin = 1e12

fitmethod = opt.minimize

class PoissonPosterior(object):

    def __init__(self, x, y, func):
        self.x = x
        self.y = y
        
        self.func = func

        return
        
    def loglikelihood(self, pars):

        model_counts = self.func(self.x, *pars)

        loglike = - np.sum(model_counts)\
        + np.sum(self.y * np.log(abs(model_counts)))\
        - np.sum(sp.special.gammaln(self.y + 1))

        if np.isnan(loglike):
           loglike = -logmin
        elif loglike == np.inf:
           loglike = -logmin
        
        return loglike

    def logposterior(self, pars):
        lpost = self.loglikelihood(pars)
        return lpost 
    
    def __call__(self, pars, neg=False):
        lpost = self.logposterior(pars) 

        ## the negative switch makes sure you can call both
        ## the log-likelihood, and the negative log-likelihood
        if neg == True:
            return -lpost
        else:
            return lpost

## set neg=True for negative log-likelihood
neg = True

#%%

R0 = 2.5
psi0 = 1
beam0 = 0
alpha0 = 0
chi0 = 0

P = 6
t0 = 1.5 * P
s0 = 1
bg0 = 3

bw = 1 / 100
rate0 = bg0 / bw 

sig4 = 0.997
y0 = poisson.ppf(sig4,bg0)

# burst duration: short: 0, medium: 1, long: 2)
burst_nr = 2

Dt_list = [15/100,3/2,3]

A0 = C1(s0,Dt_list[burst_nr])
N0 = ceil(Ncounts(y0,A0,s0,bg0,bw,Dt_list[burst_nr]))

t_max = 3 * P
N_bins = int(t_max / bw)
print(N_bins)

phi = np.linspace(0,2 * pi,int(1e5))
f = interpolate.interp1d(phi,norm_LBS(phi,R0,psi0,beam0,alpha0,chi0))
f_max = max(f(phi))

pars = [t0,A0,s0,N0,rate0]

toa_burst = []
counter_tot = 0

while counter_tot < N0:  
            
    toa = skewed_Laplace_ppf(np.random.rand(),*pars[:3])      
    phoa =  2 * pi * (toa % P) / P 

    r = np.random.rand()
    prob = f(phoa) / f_max  

    if r < prob:          
        toa_burst.append(toa)
            
    counter_tot += 1
    
toa_bg = np.random.uniform(0,t_max,int(N_bins * bg0))
toa_tot = np.sort(np.concatenate((toa_burst,toa_bg),axis=0))

N_in = len(toa_burst)
tot_counts = len(toa_tot)

t = np.linspace(0,t_max,int(1e4))
resultant = f(2 * pi * (t % P) / P) * (burst_model_pdf(t,*pars) - rate0) * bw / f_max + bg0

def N_in_func(t,dt):
    func = lambda t: f(2 * pi * (t % P) / P) * (burst_model_pdf(t,*pars) - rate0) * bw / f_max + bg0
    N_int = (sp.integrate.simps(func(t),t) - dt * bg0) / bw
    return ceil(N_int)

# toa_tot, t_max, bw, A0, s0, N_in

x_edges = np.linspace(0,t_max,int(t_max / bw) + 1,endpoint=True)
y, x_edges = np.histogram(toa_tot,x_edges)
x = x_edges[:-1] + x_edges[1] / 2

x_check = x
y_check = y

## burst identification algorithm

# running mean window
#NP = [poisson.pmf(y_i,bg0) * len(y) for y_i in y]
#indx_sig = np.where(np.array(NP) <= 0.01)[0]

# The above significance level is always at 14, so we might as well...

sig_level = 14

threshold = 3.4
incr_frac = 100

indx_sig = np.where(y >= sig_level)[0]

x_sig = x_check[indx_sig]
y_sig = y_check[indx_sig]

#j = 0
while len(y_sig) > 0:
    
    indx_y_sig_max = np.argmax(y_sig)
    
    t0_sig_max = x_sig[indx_y_sig_max]
    
    Dt_interval = 1
    
    interval_p = np.array(np.where((x_check >= t0_sig_max) & (x_check <= t0_sig_max + Dt_interval))[0])
    interval_m = np.array(np.where((x_check <= t0_sig_max) & (x_check >= t0_sig_max - Dt_interval))[0])
        
    interval_p_mean = np.mean(y_check[interval_p])
    interval_m_mean = np.mean(y_check[interval_m])
    
    incr_p = 1
    while interval_p_mean > threshold:
        interval_p = np.array(np.where((x_check >= t0_sig_max + incr_p / incr_frac) & (x_check <= t0_sig_max + Dt_interval + incr_p / incr_frac))[0])
        interval_p_mean = np.mean(y_check[interval_p])
        incr_p += 1
    if len(interval_p) < 1:
        indx_t_out = -1
    else:
        indx_t_out = interval_p[-1]    
    t_out = x_check[indx_t_out]

    incr_m = 1
    while interval_m_mean > threshold:
        interval_m = np.array(np.where((x_check <= t0_sig_max - incr_m / incr_frac) & (x_check >= t0_sig_max - Dt_interval - incr_m / incr_frac))[0])
        interval_m_mean = np.mean(y_check[interval_m])
        incr_m += 1
    if len(interval_m) < 1:
        indx_t_in = 0
    else:
        indx_t_in = interval_m[0]
    t_in = x_check[indx_t_in]
    
    indx_pdf = np.where((x_check > t_in) & (x_check < t_out))[0]

    x_fit = x_check[indx_pdf]
    y_fit = y_check[indx_pdf]
    
    t_int = np.linspace(t_in,t_out,int(1e4))
    dt_int = t_out - t_in
    
    N_in_calc = N_in_func(t_int,dt_int)
   
    initial_pars = [t0_sig_max,A0,s0,N_in_calc]
        
    pl = PoissonPosterior(x_fit, y_fit / bw, lambda t,t0,A,s,N: burst_model_pdf(t,t0,A,s,N,rate0))                      
    res = fitmethod(pl, initial_pars, args=(neg,),method = 'Nelder-Mead')
    popt = res.x  
    
#    np.savetxt('test_lc_Daniela/test_pars_data{}'.format(j),np.transpose([pars[:4],initial_pars,popt]))
#    np.savetxt('test_lc_Daniela/test_lc_data{}'.format(j),np.transpose([x_fit,y_fit / bw]))
#    j += 1
    
    x_m = np.linspace(0,t_max,int(1e4))
    
    parent = burst_model_pdf(x_m,*pars) * bw
    bfm = burst_model_pdf(x_fit,*popt,rate0) * bw

    plt.plot(x,y,linestyle='steps-mid',color=cSB)
    plt.plot(x_check,y_check,linestyle='steps-mid',color='k')
    plt.plot(x_sig,y_sig,'o',color='magenta')

    plt.plot(x_m,parent,color='DimGrey')
    plt.plot(x_fit,bfm,color='Orange')
    
    plt.xlim(t_in,t_out)
    plt.ylim(0,7/6 * max(y_sig))
    
    plt.axhline(sig_level,linestyle='dashed',color='magenta')
    plt.plot(t,resultant,'r')
    
    plt.show()
    
    indx_tot = np.arange(len(x_check)).tolist()
    indx_inverse_interval = indx_tot[:indx_t_in] + indx_tot[indx_t_out:]
    
    x_check = x_check[indx_inverse_interval]
    y_check = y_check[indx_inverse_interval]
    
    indx_sig = np.where(y_check >= sig_level)[0]

    x_sig = x_check[indx_sig]
    y_sig = y_check[indx_sig]

    plt.plot(x,y,linestyle='steps-mid',color=cSB)
    plt.plot(x_check,y_check,linestyle='steps-mid',color='k')
    
    plt.plot(x_m,parent,color='DimGrey')
    
    plt.axvline(t_in,linestyle='dotted',color='r')
    plt.axvline(t_out,linestyle='dotted',color='r')
    plt.axhline(sig_level,linestyle='dashed',color='magenta')
    plt.plot(t,resultant,'r')
    plt.xlim(0,t_max)
    
    plt.show()
     
#%%

if not os.path.isdir('RUN1_out'):
       os.makedirs('RUN1_out')

def RUN1(R,psi,beam,alpha,chi,P,DT,N_bursts,s,bg):
    
    Ch2_list = []
    
    sig_level = 14
    threshold = 3.5
    incr_frac = 100
    
    # generate the appropriate LBS
    phi = np.linspace(0,2 * pi,int(1e5))
    f = interpolate.interp1d(phi,norm_LBS(phi,R,psi,beam,alpha,chi))
    f_max = max(f(phi))    
    
    sig_burst = 0
    
    # define lightcurve properties
    bw = 0.005
    t_max = 3 * P
    Nbins = int(t_max / bw)
    rate = bg / bw
    
    A = C1(s,DT)
    N = int(ceil(Ncounts(sig_level,A,s,bg,bw,DT)))

    output_pars = []
    tot_phoa = []
        
    # burst storm loop
    for i in range(N_bursts):
        
        # set burst t0 (drawn from uniform dist.)
        t0 = np.random.uniform(0,P) + P
        
        toa_burst = []
        
        tau_rise = s / A
        tau_decay = 1 / (s * A)
        
        # define limits of the time domain where the data is to be fit
        t_in = t0 - 5 * tau_rise
        t_out = t0 + 5 * tau_decay
        
        # accept-reject method for burst counts
        counter_tot = 0
        while counter_tot < N:  
            
            # draw a time-of-arrival from the skewed Laplace point percent function
            toa = skewed_Laplace_ppf(np.random.rand(),t0,A,s)      
            phoa = (toa % P) * 2 * pi / P
            
            # generate random number (0,1) and the phase occurrence probability
            r = np.random.rand()
            prob = f(phoa) / f_max
            
            # if the phase occurrence probability is larger than a random number, keep the toa
            if r < prob:          
                toa_burst.append(toa)
            
            counter_tot += 1
        
        N_in = len(toa_burst)

        if N_in == 0:
            continue
        
        pars = [t0,A,s,N,rate]
        
        t = np.linspace(0,t_max,int(1e4))
        resultant = f(2 * pi * (t % P) / P) * (burst_model_pdf(t,*pars) - rate) * bw / f_max + bg

        def N_in_func(t,dt):
            func = lambda t: f(2 * pi * (t % P) / P) * (burst_model_pdf(t,*pars) - rate0) * bw / f_max + bg0
            N_int = (sp.integrate.simps(func(t),t) - dt * bg) / bw
            return ceil(N_int)
        
        toa_bg = np.random.uniform(0,t_max,Nbins * bg)
        toa_tot = np.concatenate((toa_burst,toa_bg),axis=0)
        
        x_edges = np.linspace(0,t_max,int(t_max / bw) + 1,endpoint=True)
        y, x_edges = np.histogram(toa_tot,x_edges)
        x = x_edges[:-1] + x_edges[1] / 2

        x_check = x
        y_check = y

        ## burst identification algorithm

        indx_sig = np.where(y >= sig_level)[0]

        x_sig = x_check[indx_sig]
        y_sig = y_check[indx_sig]
        
        while len(y_sig) > 0:
            
            indx_y_sig_max = np.argmax(y_sig)
    
            t0_sig_max = x_sig[indx_y_sig_max]
    
            Dt_interval = 1

            interval_p = np.array(np.where((x_check >= t0_sig_max) & (x_check <= t0_sig_max + Dt_interval))[0])
            interval_m = np.array(np.where((x_check <= t0_sig_max) & (x_check >= t0_sig_max - Dt_interval))[0])
        
            interval_p_mean = np.mean(y_check[interval_p])
            interval_m_mean = np.mean(y_check[interval_m])
    
            incr_p = 1
            while interval_p_mean > threshold:
                interval_p = np.array(np.where((x_check >= t0_sig_max + incr_p / incr_frac) & (x_check <= t0_sig_max + Dt_interval + incr_p / incr_frac))[0])
                interval_p_mean = np.mean(y_check[interval_p])
                incr_p += 1
            if len(interval_p) < 1:
                indx_t_out = -1
            else:
                indx_t_out = interval_p[-1]   
            t_out = x_check[indx_t_out]

            incr_m = 1
            while interval_m_mean > threshold:
                interval_m = np.array(np.where((x_check <= t0_sig_max - incr_m / incr_frac) & (x_check >= t0_sig_max - Dt_interval - incr_m / incr_frac))[0])
                interval_m_mean = np.mean(y_check[interval_m])
                incr_m += 1
            if len(interval_m) < 1:
                indx_t_in = 0
            else:
                indx_t_in = interval_m[0]
            t_in = x_check[indx_t_in]
    
            indx_pdf = np.where((x_check > t_in) & (x_check < t_out))[0]

            x_fit = x_check[indx_pdf]
            y_fit = y_check[indx_pdf]
            
            t_int = np.linspace(t_in,t_out,int(1e4))
            dt_int = t_out - t_in
    
            N_in_calc = N_in_func(t_int,dt_int)

            ### initial guess 'A' still depends on input data ###

            initial_pars =  [t0_sig_max,A,s,N_in_calc]
            
            pl = PoissonPosterior(x_fit, y_fit / bw, lambda t,t0,A,s,N: burst_model_pdf(t,t0,A,s,N,rate0))                      
            res = fitmethod(pl, initial_pars, args=(neg,),method = 'Nelder-Mead')
            popt = res.x   
            
            y_fit_is0_indx = np.where(y_fit == 0)[0]
            y_fitn0 = np.delete(y_fit,y_fit_is0_indx)
            x_fitn0 = np.delete(x_fit,y_fit_is0_indx)
    
            residuals = np.array(abs(y_fitn0 - burst_model_pdf(x_fitn0,*popt,rate0) * bw))
            errors = np.array(sqrt(y_fitn0))
            dof = len(y_fitn0) - len(popt)

            sum_i = (residuals/errors)**2
            
            sum_i[np.isinf(sum_i)] = 0
            rCh2 = sum(sum_i) / dof
            
#            x_m = np.linspace(0,t_max,int(1e4))
#    
#            parent = burst_model_pdf(x_m,*pars) * bw
#            bfm = burst_model_pdf(x_fit,*popt,rate0) * bw
#
#            plt.plot(x,y,linestyle='steps-mid',color=cSB)
#            plt.plot(x_check,y_check,linestyle='steps-mid',color='k')
#            plt.plot(x_sig,y_sig,'o',color='magenta')
#
#            plt.plot(x_m,parent,color='SlateGrey')
#            plt.plot(x_fit,bfm,color='Orange')
#            plt.plot(t,resultant,'r')
#    
#            plt.xlim(t_in,t_out)
#    
#            plt.axhline(sig_level,linestyle='dashed',color='magenta')
#    
#            plt.show()
    
            indx_tot = np.arange(len(x_check)).tolist()
            indx_inverse_interval = indx_tot[:indx_t_in] + indx_tot[indx_t_out:]
    
            x_check = x_check[indx_inverse_interval]
            y_check = y_check[indx_inverse_interval]
    
            indx_sig = np.where(y_check >= sig_level)[0]

            x_sig = x_check[indx_sig]
            y_sig = y_check[indx_sig]

#            plt.plot(x,y,linestyle='steps-mid',color=cSB)
#            plt.plot(x_check,y_check,linestyle='steps-mid',color='k')
#            plt.plot(t,resultant,'r')
#            
#            plt.xlim(0,t_max)
#    
#            plt.axvline(t_in,linestyle='dotted',color='r')
#            plt.axvline(t_out,linestyle='dotted',color='r')
#            plt.axhline(sig_level,linestyle='dashed',color='magenta')
#    
#            plt.show()

            if (rCh2 > 1.5 and rCh2 < 0.67):
                Ch2_list.append([i,rCh2])
                print("Burst fit number {} might be crap!".format(i+1))
                continue
            
            t0_bfpar, A_bfpar, s_bfpar, N_bfpar = popt
            Dt0 = popt[0] - t0
            
            ph0 = (t0 - P) / P
            ph0_bfpar = (t0_bfpar - P) / P
            
            output_pars_i = [N_in,t0,t0_bfpar,Dt0,A_bfpar,s_bfpar,N_bfpar,ph0,ph0_bfpar]
            
            output_pars.append(output_pars_i)
            
            sig_burst += 1
            
        phoa_tot_i = [(toa_i % P) * 2 * pi / P for toa_i in toa_tot]
        tot_phoa += phoa_tot_i 
        
        bursts_done = i / (N_bursts - 1)
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(bursts_done * 50), bursts_done * 100), end="", flush=True)
    
    # define the significant burst ratio
    q = sig_burst / N_bursts
    
    #print('input burst duration (Dt) = {:.2} seconds'.format(burst_duration))
    print('\nfraction of significant bursts to total bursts = {:.2}'.format(q))

    plot_data = np.transpose(output_pars)
    
    Dt0_data = plot_data[3]
    Dt0_Nbins = 200
    Dt0_bin_edges = np.linspace(-1,1,Dt0_Nbins + 1, endpoint=True)
    Dt0_counts, Dt0_bins = np.histogram(Dt0_data,Dt0_bin_edges)
    Dt0_bins = Dt0_bins[:-1] + abs(Dt0_bins[1] - Dt0_bins[0]) / 2
    
    A_data = plot_data[4]
    A_Nbins = 100
    A_bin_edges = np.linspace(0.5 * A,1.5 * A,A_Nbins + 1, endpoint=True)
    A_counts, A_bins = np.histogram(A_data,A_bin_edges)
    A_bins = A_bins[:-1] + A_bins[1] / 2
    
    s_data = plot_data[5]
    s_Nbins = 100
    s_bin_edges = np.linspace(0,4,s_Nbins + 1, endpoint=True)
    s_counts, s_bins = np.histogram(s_data,s_bin_edges)
    s_bins = s_bins[:-1] + s_bins[1] / 2
    
    N_data = plot_data[6]
#    N_Nbins = 100
#    N_bin_edges = np.logspace(3,np.log10(N),N_Nbins + 1, endpoint=True)
#    N_counts, N_bins = np.histogram(plot_data[6],N_bin_edges)
#    N_bins = N_bins[:-1] + N_bins[1] / 2

#    DT_burst(ysig,A,s,N,b,bw)
    
    DT_data = DT_burst(9,A_data,s_data,N_data,bg,bw)
    DT_data[np.isinf(DT_data)] = 0
    DT_data[np.isnan(DT_data)] = 0
    DT_Nbins = 100
    DT_bin_edges = np.linspace(0,P,DT_Nbins + 1, endpoint=True)
    DT_counts, DT_bins = np.histogram(DT_data,DT_bin_edges)
    DT_bins = DT_bins[:-1] + DT_bins[1] / 2

    plt.plot(Dt0_bins,Dt0_counts,linestyle='steps-mid',color=cSB)
    plt.axvline(0,color='k',linestyle='--')
    plt.xlim(-0.15,0.15)
    plt.xlabel(r'$\Delta t_0\ [\rm{seconds}]$')
    plt.ylabel(r'$dN/d(\Delta t_0)$')
    plt.title(r'$\rm{Peak\ time\ offset\ distribution}$')
    plt.grid(True,alpha=0.5)
    plt.show()
    
    plt.plot(A_bins,A_counts,linestyle='steps-mid',color=cSB)
    plt.axvline(A,color='k',linestyle='--',label=r'$A_0$')
    plt.xlabel(r'$A$')
    plt.ylabel(r'$dN/dA$')
    plt.title(r'$\rm{Amplitude\ parameter\ distribution}$')
    plt.legend()
    plt.grid(True,alpha=0.5)
    plt.show()

    plt.plot(s_bins,s_counts,linestyle='steps-mid',color=cSB)
    plt.axvline(s,color='k',linestyle='--',label=r'$s_0$')
    plt.xlim(0,s + 1)
    plt.xlabel(r'$s$')
    plt.ylabel(r'$dN/ds$')
    plt.title(r'$\rm{Skew\ parameter\ distribution}$')
    plt.legend()
    plt.grid(True,alpha=0.5)
    plt.show()
    
#    plt.plot(N_bins,N_counts,linestyle='steps-mid',color=cSB)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlabel(r'$Fluence$')
#    plt.ylabel(r'$dF/dN$')
#    plt.title(r'$\rm{Fluence\ parameter\ (N)\ distribution}$')
#    plt.grid(True,alpha=0.5)
#    plt.show()

    plt.plot(DT_bins,DT_counts,linestyle='steps-mid',color=cSB)
    plt.axvline(DT,color='k',linestyle='--',label=r'$\Delta T_0$')
    plt.xlim(0,P)
    plt.xlabel(r'$\Delta T$')
    plt.ylabel(r'$dN/d(\Delta T)$')
    plt.title(r'$\rm{Burst\ duration\ distribution}$')
    plt.legend()
    plt.grid(True,alpha=0.5)
    plt.show()

    ph_Nbins = 20
    ph_bin_edges = np.linspace(0,2 * pi,ph_Nbins + 1,endpoint=True)
    ph_counts, ph_bins = np.histogram(tot_phoa,ph_bin_edges)
    ph_counts_err = np.sqrt(ph_counts)
    ph_bw = ph_bins[1]
    
    ph_bins = (ph_bins[:-1] + ph_bw / 2)/(2 * pi)
    ph_counts = ph_counts / ph_bw / len(tot_phoa)
    ph_counts_err = ph_counts_err / ph_bw / len(tot_phoa)
    
    ph0_bfpar = [(t0_bfpar_i % P) / P for t0_bfpar_i in plot_data[2]]     
    ph0_Nbins = 20
    ph0_bin_edges = np.linspace(0,1,ph0_Nbins + 1, endpoint=True)    
    ph0_counts, ph0_bins = np.histogram(ph0_bfpar,ph0_bin_edges)
    ph0_bw = ph0_bins[1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(6,8))
    
    # plot: applied LBS vs phase
    ax1.plot(phi/(2 * pi),f(phi)/f_max,color='k')
    ax1.set_ylabel(r'$\kappa(\phi)$')
    ax1.set_title(r'$R={0},\ \psi={1}^{{\circ}},\ \rm{{beam}}={2},\ \alpha={3}^{{\circ}},\ \chi={4}^{{\circ}}$'.format(R,psi,beam,alpha,chi))
    ax1.grid(True,alpha=0.5)
    
    # plot: count density vs phase (only contains the counts of the cycles containing significant bursts)
    ax2.errorbar(ph_bins,ph_counts,yerr = ph_counts_err,color='k',linestyle='steps-mid')
    ax2.set_ylabel(r'$\rm{Count\ density}$')
    ax2.set_title(r'$\rm{{Counts\ per\ burst = {0},\ \ Background\ counts\ per\ cycle = {1}}}$'.format(N,int(Nbins/3)))
    ax2.grid(True,alpha=0.5)
    
    # plot: best-fit phi occurrence vs phase
#    ax3.plot((ph0_bins[:-1] + ph0_bw / 2),ph0_counts_cdf,color='Crimson',linestyle='steps-mid')
    ax3.plot((ph0_bins[:-1] + ph0_bw / 2),ph0_counts,color=cSB,linestyle='steps-mid')
    ax3.set_xlabel(r'$\phi/2\pi$')
    ax3.set_ylabel(r'$\phi_0\ \rm{distribution}$')
    ax3.set_title(r'$N_{{\rm{{bursts}}}}^{{\rm{{tot}}}}={0},\ N_{{\rm{{bursts}}}}^{{\rm{{sig}}}}={1},\ q = {2:.2}$'.format(N_bursts,len(ph0_bfpar),q))
    ax3.grid(True,alpha=0.5)
    
    plt.subplots_adjust(hspace=0.2)
    plt.show()

    np.savetxt('RUN1_out/test/chi',Ch2_list)    
    
    ## OUTPUT THE SIMULATED DATA
#    np.savetxt('RUN1_out/test/pars_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}_{9}'.format(R,psi,beam,alpha,chi,P,DT,N_bursts,s,bg),output_pars)
#    np.savetxt('RUN1_out/test/hist_s_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}_{9}'.format(R,psi,beam,alpha,chi,P,DT,N_bursts,s,bg),np.transpose([s_bins,s_counts]))
#    np.savetxt('RUN1_out/test/hist_Dt0_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}_{9}'.format(R,psi,beam,alpha,chi,P,DT,N_bursts,s,bg),np.transpose([Dt0_bins,Dt0_counts]))  
#    np.savetxt('RUN1_out/test/hist_ph0_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}_{9}'.format(R,psi,beam,alpha,chi,P,DT,N_bursts,s,bg),np.transpose([ph_bins,ph_counts,ph_counts_err]))
    

#%%
from time import strftime, localtime
time_in = strftime('%H:%M:%S', localtime())

R = 2.5
psi = 1
beam = 0
alpha = 90
chi = 90
P = 6

DT = Dt_list[2]
N_bursts = 1000

s = 1
bg = 3

RUN1(R,psi,beam,alpha,chi,P,DT,N_bursts,s,bg)

print(time_in)
print(strftime('%H:%M:%S', localtime()))

#%%

data = np.loadtxt('RUN1_out/test/chi')

bins = np.linspace(0,2,100)
counts, bins = np.histogram(data,bins)
plt.plot(bins[:-1],counts)
plt.xlim(0,2)

#%%
length = len(data)
ch_sort = sorted(np.transpose(data)[1])

#%%
#from time import strftime, localtime
#time_in = strftime('%H:%M:%S', localtime())
#
#R = 2.5
#psi = 1
#beam = 0
#P = 6
#N_bursts = 10000
#s = 1
#bg = 3
#
#alpha_l = [90,45]
#chi_l = [90,45]
#
#for i in range(3):
#    for j in range(2):
#        RUN1(R,psi,beam,alpha_l[j],chi_l[j],P,N_bursts,A_l[i],s,burst_counts_l[i],bg)
#
#print(time_in)
#print(strftime('%H:%M:%S', localtime()))

#%%
import astropy.modeling.models as models
from astropy.modeling.models import custom_model

from stingray.modeling.posterior import PoissonLogLikelihood, PoissonPosterior
from stingray.modeling.parameterestimation import ParameterEstimation

@custom_model
def burst_model_pdf(t,t0,A,s,N,bg_rate):
    return N * skewed_Laplace_pdf(t,t0,A,s) + bg_rate

data = np.transpose(np.loadtxt('./test_lc_Daniela/test_lc_data0'))
data_par = np.transpose(np.loadtxt('./test_lc_Daniela/test_pars_data0'))

x, counts = data
par_input, par_init, par_bf = data_par

#%%

mm = burst_model_pdf(t0=1,A=1,s=1,N=2000,bg_rate=300)

#%%

mm.param_names






























