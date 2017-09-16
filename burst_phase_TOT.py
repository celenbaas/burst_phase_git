import os

os.chdir('/Users/DexterIII/Documents/Werk/PhD/research/projects/burst_phase/work_folder')

#%%

import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy.stats import poisson
import matplotlib.pyplot as plt

from numpy import pi,radians,arcsin,cos,sin,sqrt,arccos,exp,log
from scipy import integrate, interpolate
    
def A(r):
    return 1 - 1 / r
    
#%%
if not os.path.isdir('theta_star_inf_out'):
       os.makedirs('theta_star_inf_out')

def theta_star_inf(R,N_u,N_b):
    
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
    
    np.savetxt('theta_star_inf_out/theta_star_inf_R_{0}_Nu_{1}_Nb_{2}'.format(R,logN_u,logN_b),theta_star_data,delimiter=' ')

#%%

#theta_star_inf(2.5,1e5,1e5)

#%%

R = 2.5
N_u = 1e5
N_b = 1e5

b_max = R * A(R)**(-1/2)

b, theta_star = np.transpose(np.loadtxt('theta_star_inf_out/theta_star_inf_R_{0}_Nu_{1}_Nb_{2}'.format(R,int(np.log10(N_u)),int(np.log10(N_b)))))
theta_star_b = interpolate.interp1d(b,theta_star)

cSB = 'SteelBlue'

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(b,theta_star,color = cSB) 
ax.axvline(b_max * (1-1e-6),color='k',linestyle='dashed')
ax.set_xlabel(r'$b$')
ax.set_ylabel(r'$ \theta_*(b) $')
ax.set_title(r'$R = {0}$'.format(R))
plt.show()

#%%

##%%
#
#if not os.path.isdir('Dt_star_inf_out'):
#       os.makedirs('Dt_star_inf_out')
#
#def Dt_star_inf(R,N_b):
#    
#    b_max = R * A(R)**(-1/2)
#    
#    def integral(b):
#        
#        intgrand = lambda r: 1 / (1 - 1 / r) * ((1 - (b / r)**2 * (1 - 1 / r))**(-1/2) - 1)
#        intgrated  = integrate.quad(intgrand,R,np.inf)[0]
#        return intgrated
#    
#    b_array = np.linspace(0,b_max,int(N_b),endpoint=True)
#    Dt_star_array = [integral(b_i) for b_i in b_array]
#    Dt_star_data = np.transpose([b_array,Dt_star_array])
#    
#    logN_b = int(np.log10(N_b))
#    
#    np.savetxt('Dt_star_inf_out/Dt_star_inf_R_{0}_Nb_{1}'.format(R,logN_b),Dt_star_data,delimiter=' ')    
#
##%%
#
#Dt_star_inf(2.5,1e5)
#
##%%
#
#R = 2.5
#N_b = 1e5
#
#b_max = R * A(R)**(-1/2)
#
#b, Dt_star = np.transpose(np.loadtxt('Dt_star_inf_out/Dt_star_inf_R_{0}_Nb_{1}'.format(R,int(np.log10(N_b)))))
#Dt_star_b = interpolate.interp1d(b,Dt_star)
#
#fig, ax = plt.subplots(1,1,figsize=(6,4))
#ax.plot(b,T_star,color = cSB) 
#ax.axvline(b_max,color='k',linestyle='dashed')
#ax.set_xlabel(r'$b$')
#ax.set_ylabel(r'$\Delta T_*(b) $')
#ax.set_title(r'$R = {0}$'.format(R))
#ax.set_yscale('log')
#plt.show()

#%%

if not os.path.isdir('LBS_out'):
       os.makedirs('LBS_out')

def generate_LBS(R,psi,beam):

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

    np.savetxt('LBS_out/LBS_R_{0}_psi_{1}_beam_{2}_test'.format(R,psi,beam),np.transpose([theta_0_array,I_obs]))
    #plt.plot(theta_0_array,I_obs)
    #plt.show()

#%%

#generate_LBS(2.5,1,0)

#%%

R = 2.5
psi = 1
beam = 0

theta, LBS = np.transpose(np.loadtxt('LBS_out/LBS_R_{0}_psi_{1}_beam_{2}'.format(R,psi,beam)))
LBS_theta = interpolate.interp1d(theta,LBS)

indx_terminator = np.where(LBS <= 1e-7)[0][0]

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(theta / pi,LBS_theta(theta),color=cSB) 
ax.axvline(theta[indx_terminator] / pi,color='k',linestyle='dashed')
ax.set_xlabel(r'$\theta_0 / \pi$')
ax.set_ylabel(r'$ \kappa(\theta_0) $')
ax.set_title(r'$R = {0}$'.format(R))
plt.show()

#%%

def norm_LBS(phi,R,psi,beam,alpha,chi):
    
    theta, LBS = np.transpose(np.loadtxt('LBS_out/LBS_R_{0}_psi_{1}_beam_{2}'.format(R,psi,beam)))
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
ax.axvline(theta[indx_terminator] / (2 * pi),color='k',linestyle='dashed')
ax.axvline(1 - theta[indx_terminator + 1] / (2 * pi),color='k',linestyle='dashed')
ax.set_xlim(0,1)
ax.set_xlabel(r'$\phi/2\pi$')
ax.set_ylabel(r'$ \rm{Normalized\ LBS} $')
ax.set_title(r'$R = {0}$'.format(R))
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
    t_t0 = lambda t: A / (s + 1 / s) * exp(A * (t - t0) / s)
    t0_t = lambda t: A / (s + 1 / s) * exp(- A * s * (t - t0))
    return np.piecewise(t, [t < t0, t >= t0], [t_t0,t0_t]) 

# skewed Laplace cumulative density function
def skewed_Laplace_cdf(t,t0,A,s):    
    t_t0 = lambda t: s**2 / (1 + s**2) * exp(A * (t - t0) / s)
    t0_t = lambda t: 1 - 1 / (1 + s**2) * exp(- A * s * (t - t0))
    return np.piecewise(t, [t <= t0, t > t0], [t_t0,t0_t]) 
        
# skewed Laplace point percent function
def skewed_Laplace_ppf(q,t0,A,s):
    XminM = lambda q: t0 + log((1 + s**2) * q / s**2) * s / A
    XplusM = lambda q: t0 - log((1 - q) * (1 + s**2)) / s / A
    return np.piecewise(q, [t0 > XminM(q), t0 < XplusM(q)], [XminM(q),XplusM(q)])

#%%

def tmin(y,t0,A,s):
    return s / A * np.log(y * (1 + s**2) / s) + t0

def tplus(y,t0,A,s):
    return (- 1) / (A * s) * np.log(y * (1 + s**2)) + t0
        
#%%

# my model is the skewed Laplace probability density function with normalisation *N* and background *bg*
def burst_model_pdf(t,t0,A,s,N,bg_rate):
    return N * skewed_Laplace_pdf(t,t0,A,s) + bg_rate

def burst_model_cdf(t,t0,A,s,N,bg_rate,offset):
    return N * skewed_Laplace_cdf(t,t0,A,s) + bg_rate * t + offset

def Dt_burst(ysig,A,s,N,b,bw):
    return - (1 + s**2)/(A * s) * log( (s + 1 / s)/(N * A * bw) * (ysig - b))

#def burst_model_ppf(q,t0,A,s,N,bg_rate):
#    XminM = lambda q: q / bg_rate - (s * sp.special.lambertw((A * np.exp((A * (q - bg_rate * t0)) / (bg_rate * s)) * N * s) / (bg_rate * (1 + s**2)))) / A
#    XplusM = lambda q: (- N + q) / bg_rate + sp.special.lambertw((A * np.exp((A * s * (N - q + bg_rate * t0)) / bg_rate) * N * s) / (bg_rate * (1 + s**2))) / (A * s)
#    return np.piecewise(q, [t0 > XminM(q), t0 < XplusM(q)], [XminM(q),XplusM(q)])
    
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
R0 = 1000.0
psi0 = 1
beam0 = 0
alpha0 = 90
chi0 = 90

phi = np.linspace(0,2 * pi,int(1e5))
f = interpolate.interp1d(phi,norm_LBS(phi,R0,psi0,beam0,alpha0,chi0))
f_max = max(f(phi))

P = 6
t0 = 1.2 * P
A0 = 1.0747790879610903
s0 = 1
burst_counts0 = 5000
bg0 = 3
bw = 0.005
t_max = 3 * P
rate = bg0 / bw 
N_bins = int(t_max / bw)

pars = [t0,A0,s0]

tau_rise = s0 / A0
tau_decay = 1 / (s0 * A0)

t_fac = 5
t_in = t0 - tau_rise * t_fac / s0
t_out = t0 + tau_decay * t_fac 

toa_burst = []
counter_tot = 0
while counter_tot < burst_counts0:  
            
    # draw a time-of-arrival from the skewed Laplace point percent function
    toa = skewed_Laplace_ppf(np.random.rand(),t0,A0,s0)      
    phoa = (toa % P) * 2 * pi / P
            
    # generate random number (0,1) and the phase occurrence probability
    r = np.random.rand()
    prob = f(phoa) / f_max  
            
    #print(r,prob)
            
    # if the phase occurrence probability is larger than a random number, keep the toa
    if r < prob:          
        toa_burst.append(toa)
            
    counter_tot += 1
    
toa_bg = np.random.uniform(0,t_max,N_bins * bg0)
toa_tot = np.sort(np.concatenate((toa_burst,toa_bg),axis=0))
        
N_in = len(toa_burst)

tot_counts = len(toa_tot)

x_cdf = np.sort(toa_tot)
y_cdf = np.arange(tot_counts)

indx_cdf = np.where( (x_cdf > t_in) & (x_cdf < t_out) )[0]

x_cdf_fit = x_cdf[indx_cdf]
y_cdf_fit = y_cdf[indx_cdf]

offset = y_cdf_fit[0] - rate * t_in

pars_cdf = pars + [burst_counts0,rate,offset]
pl_cdf = PoissonPosterior(x_cdf_fit, y_cdf_fit, burst_model_cdf)                 
res_cdf = fitmethod(pl_cdf, pars_cdf, args=(neg,),method = 'Nelder-Mead')
popt_cdf = res_cdf.x

x_pdf_edges = np.linspace(0,t_max,int(t_max / bw) + 1,endpoint=True)
y_pdf, x_pdf_edges = np.histogram(toa_tot,x_pdf_edges)
x_pdf = x_pdf_edges[:-1] + x_pdf_edges[1] / 2

#from math import factorial as fac

## burst identification algorithm

# running mean window
N_window = int(P / (2 * bw))

mu_lst = []
NP_lst = []

sig2 = 0.95
y_sig0 = poisson.ppf(sig2,bg0)
 
for i in range(N_window,len(y_pdf) - N_window):
    y_pre_i = y_pdf[:i - N_window]
    y_post_i = y_pdf[i + N_window:]
    
    mu_i = np.mean(np.concatenate((y_pre_i,y_post_i), axis=0))  
    mu_lst.append(mu_i)
    
    NP_i = poisson.pmf(y_pdf[i],mu_i) * (len(y_pdf) - 2 * N_window)  
    NP_lst.append(NP_i)
    
sig_indx = np.where(np.array(NP_lst) <= 0.01)[0]

bins_int = x_pdf[N_window:-N_window]
sig_time = bins_int[sig_indx]

counts_int = y_pdf[N_window:-N_window]
sig_counts = counts_int[sig_indx]

plt.plot(x_pdf,y_pdf,linestyle='steps-mid',color='k')
plt.plot(sig_time,sig_counts,'bo')
plt.axhline(y_sig0)
plt.xlim(0,t_max)

indx_pdf = np.where( (x_pdf > t_in) & (x_pdf < t_out) )[0]

x_pdf_fit = x_pdf[indx_pdf]
y_pdf_fit = y_pdf[indx_pdf]

pars_pdf = pars + [burst_counts0,rate]
pl_pdf = PoissonPosterior(x_pdf_fit, y_pdf_fit / bw, burst_model_pdf)                      
res_pdf = fitmethod(pl_pdf, pars_pdf, args=(neg,),method = 'Nelder-Mead')
popt_pdf = res_pdf.x

x = np.linspace(0,t_max,tot_counts)

parent_cdf = burst_model_cdf(x,*pars_cdf)
bfm_cdf = burst_model_cdf(x_cdf_fit,*popt_cdf)

parent_pdf = burst_model_pdf(x,*pars_pdf)
bfm_pdf = burst_model_pdf(x_pdf_fit,*popt_pdf)
bfm_pdf_cdfpar = burst_model_pdf(x_pdf_fit,*popt_cdf[:-1])

print(*pars_pdf)

if res_cdf.success:
    fit_cdf = 'success'
else: fit_cdf = 'fail'

if res_pdf.success:
    fit_pdf = 'success'
else: fit_pdf = 'fail'
    
cSG = 'SlateGrey' 

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8), sharex = True)

x_lowl = t_in - P
x_upl = t_out + P

y_lowl =  burst_model_cdf(x_lowl,*pars_cdf) / tot_counts
y_upl = burst_model_cdf(x_upl,*pars_cdf) / tot_counts

ax1.axvline(t0,linestyle='--',color='k')
ax1.plot(x_cdf,y_cdf / tot_counts,linestyle='steps-mid',label='data',color='k')
ax1.plot(x,parent_cdf / tot_counts,label='parent',color=cSG,lw=2,linestyle='--')
ax1.plot(x_cdf_fit,bfm_cdf / tot_counts,label='bfm cdf',color='Crimson')
ax1.legend(loc=4, prop={'size': 10})
ax1.set_xlim(x_lowl,x_upl)
ax1.set_ylim(y_lowl,y_upl)
ax1.set_title(r'$\rm{{Normalized\ cumulative\ burst\ profile,\ fit: {0}}}$'.format(fit_cdf))
ax1.grid(True,alpha=0.5)

ax2.axvline(t0,linestyle='--',color='k')
ax2.plot(x_pdf,y_pdf,linestyle='steps-mid',label='data',color='k')
ax2.plot(x,parent_pdf * bw,label='parent',color=cSG,lw=2,linestyle='--')
ax2.plot(x_pdf_fit,bfm_pdf * bw,label='bfm pdf',color=cSB)
#ax2.plot(x_pdf_fit,bfm_pdf_cdfpar * bw,label='bfm cdf',color='Crimson')
ax2.legend(loc=1,prop={'size': 10})
ax2.set_xlim(x_lowl,x_upl)
ax2.set_title(r'$\rm{{Burst\ profile,\ fit: {0}}}$'.format(fit_pdf))
ax2.grid(True,alpha=0.5)
plt.show()

#%%

if not os.path.isdir('RUN1_out'):
       os.makedirs('RUN1_out')

def RUN1(R,psi,beam,alpha,chi,P,N_bursts,A,s,burst_counts,bg):
    
    # generate the appropriate LBS
    phi = np.linspace(0,2 * pi,int(1e5))
    f = interpolate.interp1d(phi,norm_LBS(phi,R,psi,beam,alpha,chi))
    f_max = max(f(phi))
        
    sig_burst = 0
    
    # define lightcurve properties
    bw = 0.005
    t_max = 3 * P
    N_bins = int(t_max / bw)
    rate = bg / bw
    
    sig2 = 0.95
    y_sig = poisson.ppf(sig2,bg)
    burst_duration = Dt_burst(y_sig,A,s,burst_counts,bg,bw)

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
        while counter_tot < burst_counts:  
            
            # draw a time-of-arrival from the skewed Laplace point percent function
            toa = skewed_Laplace_ppf(np.random.rand(),t0,A,s)      
            phoa = (toa % P) * 2 * pi / P
            
            # generate random number (0,1) and the phase occurrence probability
            r = np.random.rand()
            prob = f(phoa) / f_max  
            
            #print(r,prob)
            
            # if the phase occurrence probability is larger than a random number, keep the toa
            if r < prob:          
                toa_burst.append(toa)
            
            counter_tot += 1
        
        N_in = len(toa_burst)

        if N_in == 0:
            continue
        
        ### ???? ###
        ## Fit burst profile (if burst is significant) ##
        # We calculate the duration of a burst with the same input parameters, but now
        # consisting only of N_in counts. If this burst is above the 2sigma level, i.e.
        # if its 2sigma duration Dt_sig is positive (Dt_sig > 0) we label the burst as 
        # significant and proceed to fit the burst.
        
        # This is however quite an arbitrary way to determine the significance of the 
        # burst since the values for 't0', 'a', and 's' may be different.
        
        Dt_sig = Dt_burst(y_sig,A,s,N_in,bg,bw)
       
        if Dt_sig > 0:
            
            ### ???? ###
            # As initial fit parameters I use the parameters of the input burst.
            # However this might bias the result of the fit!
            
            # define fit parameters
            pars = [t0,A,s,N_in,rate]
#            parent_pars = [t0,A,s,burst_counts,rate]
            
            # combine the burst counts with the backgrount counts
            toa_bg = np.random.uniform(0,t_max,N_bins * bg)
            toa_tot = np.concatenate((toa_burst,toa_bg),axis=0)
            #tot_counts = len(toa_tot)
            
            #x_cdf = np.sort(toa_tot)
            #y_cdf = np.arange(tot_counts)
            
            x_pdf_edges = np.linspace(0,t_max,N_bins + 1,endpoint=True)
            y_pdf, x_pdf_edges = np.histogram(toa_tot,x_pdf_edges)
            
#            fit_indx_cdf = np.where( (x_cdf > t_in) & (x_cdf < t_out) )[0]
#            x_cdf_fit = x_cdf[fit_indx_cdf]
#            y_cdf_fit = y_cdf[fit_indx_cdf]
            
            # account for the offset at t_in
#            offset = y_cdf_fit[0] - rate * t_in
            
            x_pdf = x_pdf_edges[:-1] + x_pdf_edges[1] / 2
            y_pdf = y_pdf / bw

            fit_indx_pdf = np.where( (x_pdf > t_in) & (x_pdf < t_out) )[0]
            x_pdf_fit = x_pdf[fit_indx_pdf]
            y_pdf_fit = y_pdf[fit_indx_pdf]
            
#            pl_cdf = PoissonPosterior(x_cdf_fit, y_cdf_fit, burst_model_cdf) 
            pl_pdf = PoissonPosterior(x_pdf_fit, y_pdf_fit, burst_model_pdf)
                                      
            #pars_cdf = pars + [offset]
#            parent_pars_cdf = parent_pars + [offset]
            
            #res_cdf = fitmethod(pl_cdf, pars_cdf, args=(neg,),method = 'Nelder-Mead')
#            if res_cdf.success:
#                fit_cdf = 'success'
#            else: fit_cdf = 'fail'
            
            res_pdf = fitmethod(pl_pdf, pars, args=(neg,),method = 'Nelder-Mead')
#            if res_pdf.success:
#                fit_pdf = 'success'
#            else: fit_pdf = 'fail'          
            
            ### ???? ###
            # Here we determine the best-fit parameters. However, it would be good
            # if we could determine their confidence intervals as well! The spread
            # in the final distributions may mainly be due to the fact that it will
            # become more difficult to obtain a good fit if the burst consists of 
            # less counts.

#            popt_cdf = res_cdf.x
            popt_pdf = res_pdf.x
            
            t0_bfpar, A_bfpar, s_bfpar = popt_pdf[0:3]
            Dt0_pdf = popt_pdf[0] - t0
            
            ph0 = (t0 - P) / P
            ph0_bfpar = (t0_bfpar - P) / P

            output_pars.append([N_in,t0,t0_bfpar,Dt0_pdf,A_bfpar,s_bfpar,ph0,ph0_bfpar])
            
            sig_burst += 1
            
            # here we can choose to plot every single burst: not recommended for large burst storms

#            xm = np.linspace(0,t_max,10 * N_bins + 1,endpoint=True)
#            #fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,10),sharex=True)
#            fig, ax2 = plt.subplots(1,figsize=(6,10),sharex=True)
            
#            ax1.axvline((t0-P)/P,linestyle='--',color='k')
#            
#            ax1.plot((x_cdf - P) / P,y_cdf / tot_counts,linestyle='steps-mid',label='data',color='k')
#            ax1.plot((xm - P) / P,burst_model_cdf(xm,*parent_pars_cdf) / tot_counts,label='parent',color=cSG,lw=2,linestyle='--')
#            ax1.plot((x_cdf_fit - P) / P,burst_model_cdf(x_cdf_fit,*popt_cdf) / tot_counts,label='bfm cdf',color='Crimson')   
#            ax1.set_title(r'$\rm{{Burst}} = {0}, fit.cdf:{1}, fit.pdf:{2}$'.format(sig_burst,fit_cdf,fit_pdf))
#            ax1.set_xlim(t_in,t_out)
#            ax1.set_ylim(t_in * rate / tot_counts,t_out * rate / tot_counts)
#            ax1.legend(loc=4, prop={'size': 10})
#
#            ax2.plot((x_pdf-P)/P,y_pdf,linestyle='steps-mid',label='data',color='k')
#            ax2.plot((xm-P)/P,burst_model_pdf(xm,*parent_pars),label='parent',color=cSG,lw=2,linestyle='--')
#            ax2.plot((x_pdf_fit-P)/P,burst_model_pdf(x_pdf_fit,*popt_pdf),label='bfm pdf',color=cSB)
#            #ax2.plot((x_cdf_fit-P)/P,burst_model_pdf(x_cdf_fit,*popt_cdf[:-1]),label='bfm cdf',color='Crimson')
#
#            ax2.set_xlim((t_in-P)/P-0.1,(t_out-P)/P+0.1)
#            ax2.legend(loc=1, prop={'size': 10})
#            ax2.set_xlabel(r'$\phi/2\pi$')
#            
#            plt.subplots_adjust(hspace=0.1)
#            
#            plt.figure()
        else: continue
            
        phoa_tot_i = [(toa_i % P) * 2 * pi / P for toa_i in toa_tot]
        tot_phoa += phoa_tot_i 
    
    # define the significant burst ratio
    q = sig_burst / N_bursts
    
    print('input burst duration (Dt) = {:.2} seconds'.format(burst_duration))
    print('fraction of significant bursts to total bursts = {:.2}'.format(q))

    plot_data = np.transpose(output_pars)

    A_Nbins = 100
    A_bin_edges = np.linspace(0,A+1,A_Nbins + 1, endpoint=True)
    A_counts_pdf, A_bins = np.histogram(plot_data[4],A_bin_edges)
    A_bins = A_bins[:-1] + A_bins[1] / 2
    
    s_Nbins = 100
    s_bin_edges = np.linspace(0,4,s_Nbins + 1, endpoint=True)
    s_counts_pdf, s_bins = np.histogram(plot_data[5],s_bin_edges)
    s_bins = s_bins[:-1] + s_bins[1] / 2

    Dt0_Nbins = 100
    Dt0_bin_edges = np.linspace(-1,1,Dt0_Nbins + 1, endpoint=True)
    Dt0_counts_pdf, Dt0_bins = np.histogram(plot_data[3],Dt0_bin_edges)
    Dt0_bins = Dt0_bins[:-1] + abs(Dt0_bins[1] - Dt0_bins[0]) / 2
    
    ph0_bfpar_pdf = [(t0_bfpar_i % P) / P for t0_bfpar_i in plot_data[2]] 
    
    ph0_Nbins = 20
    ph0_bin_edges = np.linspace(0,1,ph0_Nbins + 1, endpoint=True)
    
#    ph0_counts_cdf, ph0_bins = np.histogram(ph0_bfpar_cdf,ph0_bin_edges)
    ph0_counts_pdf, ph0_bins = np.histogram(ph0_bfpar_pdf,ph0_bin_edges)
    
    ph0_bw = ph0_bins[1]
    
    # plot the best-fit skewness parameter distribution
#    plt.plot(s_bins,s_counts_cdf,linestyle='steps-mid',color='Crimson')
    plt.plot(A_bins,A_counts_pdf,linestyle='steps-mid',color=cSB)
    plt.axvline(A,color='k',linestyle='--',label=r'$s_0$')
#    plt.xlim(0,s + 1)
    plt.xlabel(r'$A$')
    plt.ylabel(r'$dN/dA$')
    plt.title(r'$\rm{Amplitude\ parameter\ distribution}$')
    plt.legend()
    plt.grid(True,alpha=0.5)
    plt.show()

    # plot the best-fit skewness parameter distribution
#    plt.plot(s_bins,s_counts_cdf,linestyle='steps-mid',color='Crimson')
    plt.plot(s_bins,s_counts_pdf,linestyle='steps-mid',color=cSB)
    plt.axvline(s,color='k',linestyle='--',label=r'$s_0$')
    plt.xlim(0,s + 1)
    plt.xlabel(r'$s$')
    plt.ylabel(r'$dN/ds$')
    plt.title(r'$\rm{Skew\ parameter\ distribution}$')
    plt.legend()
    plt.grid(True,alpha=0.5)
    plt.show()
    
    # plot the difference with input and best-fit t_0 distribution
#    plt.plot(Dt0_bins,Dt0_counts_cdf,linestyle='steps-mid',color='Crimson')
    plt.plot(Dt0_bins,Dt0_counts_pdf,linestyle='steps-mid',color=cSB)
    plt.axvline(0,color='k',linestyle='--')
    plt.xlim(-0.15,0.15)
    plt.xlabel(r'$\Delta t_0\ [\rm{seconds}]$')
    plt.ylabel(r'$dN/d(\Delta t_0)$')
    plt.title(r'$\rm{Peak\ time\ offset\ distribution}$')
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(6,8))
    
    # plot: applied LBS vs phase
    ax1.plot(phi/(2 * pi),f(phi)/f_max,color='k')
    ax1.set_ylabel(r'$\kappa(\phi)$')
    ax1.set_title(r'$R={0},\ \psi={1}^{{\circ}},\ \rm{{beam}}={2},\ \alpha={3}^{{\circ}},\ \chi={4}^{{\circ}}$'.format(R,psi,beam,alpha,chi))
    ax1.grid(True,alpha=0.5)
    
    # plot: count density vs phase (only contains the counts of the cycles containing significant bursts)
    ax2.errorbar(ph_bins,ph_counts,yerr = ph_counts_err,color='k',linestyle='steps-mid')
    ax2.set_ylabel(r'$\rm{Count\ density}$')
    ax2.set_title(r'$\rm{{Counts\ per\ burst = {0},\ \ Background\ counts\ per\ cycle = {1}}}$'.format(burst_counts,int(N_bins/3)))
    ax2.grid(True,alpha=0.5)
    
    # plot: best-fit phi occurrence vs phase
#    ax3.plot((ph0_bins[:-1] + ph0_bw / 2),ph0_counts_cdf,color='Crimson',linestyle='steps-mid')
    ax3.plot((ph0_bins[:-1] + ph0_bw / 2),ph0_counts_pdf,color=cSB,linestyle='steps-mid')
    ax3.set_xlabel(r'$\phi/2\pi$')
    ax3.set_ylabel(r'$\phi_0\ \rm{distribution}$')
    ax3.set_title(r'$N_{{\rm{{bursts}}}}^{{\rm{{tot}}}}={0},\ N_{{\rm{{bursts}}}}^{{\rm{{sig}}}}={1},\ q = {2:.2}$'.format(N_bursts,len(ph0_bfpar_pdf),q))
    ax3.grid(True,alpha=0.5)
    
    plt.subplots_adjust(hspace=0.2)
    plt.show()    
    
    ## OUTPUT THE SIMULATED DATA
    np.savetxt('RUN1_out/pars_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}'.format(R,alpha,chi,P,N_bursts,A,s,burst_counts,bg),output_pars)
    np.savetxt('RUN1_out/hist_s_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}'.format(R,alpha,chi,P,N_bursts,A,s,burst_counts,bg),np.transpose([s_bins,s_counts_pdf]))
    np.savetxt('RUN1_out/hist_Dt0_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}'.format(R,alpha,chi,P,N_bursts,A,s,burst_counts,bg),np.transpose([Dt0_bins,Dt0_counts_pdf]))  
    np.savetxt('RUN1_out/hist_ph0_{0}_{1}_{2}_{3}_{4}_{5:.1f}_{6}_{7}_{8}'.format(R,alpha,chi,P,N_bursts,A,s,burst_counts,bg),np.transpose([ph_bins,ph_counts,ph_counts_err]))

#%%
from time import strftime, localtime
time_in = strftime('%H:%M:%S', localtime())

R = 1000.0
psi = 1
beam = 0
alpha = 90
chi = 90
P = 6
N_bursts = 500
A = 0.5
s = 1
burst_counts = 8000
bg = 3

RUN1(R,psi,beam,alpha,chi,P,N_bursts,A,s,burst_counts,bg)

print(time_in)
print(strftime('%H:%M:%S', localtime()))

#%%
from time import strftime, localtime
time_in = strftime('%H:%M:%S', localtime())

R = 2.5
psi = 1
beam = 0
P = 6
N_bursts = 10000
s = 1
bg = 3

A_l = [0.5373895439805452,1.0747790879610903,10.747790879610903]
burst_counts_l = [10000,5000,500]


alpha_l = [90,45]
chi_l = [90,45]

for i in range(3):
    for j in range(2):
        RUN1(R,psi,beam,alpha_l[j],chi_l[j],P,N_bursts,A_l[i],s,burst_counts_l[i],bg)

print(time_in)
print(strftime('%H:%M:%S', localtime()))