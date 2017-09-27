import matplotlib.pyplot as plt

import argparse
import numpy as np
import scipy.stats

import astropy.modeling.models as models
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import _fitter_to_model_params
import corner

from stingray.modeling.posterior import PoissonLogLikelihood, PoissonPosterior
from stingray.modeling.parameterestimation import ParameterEstimation


@custom_model
def BurstShape(x, t0=1.0, log_amplitude=0.0, tau=0.1, skew=1.0):
    """
    Exponential rise/exponential decay burst profile.

    Parameters
    ----------
    x : iterable
        The array with time stamps

    t0 : float
        peak time

    log_amplitude : float
        the logarithm of the peak amplitude of the burst

    tau : float, tau > 0
        burst rise time

    skew : float
        skewness of the burst, such that fall time = skew * tau

    """
    x = np.array(x)

    # indices of values below t0
    mask_low = (x < t0)
    # indices of values above or equal t0
    mask_high = (x >= t0)
    
    # empty array for the model flux
    profile = np.zeros_like(x)
    
    # exponentiate log-amplitude
    amplitude = np.exp(log_amplitude)
    
    # below t0: exponential rise
    profile[mask_low] = amplitude * np.exp((x[mask_low]-t0)/tau)
    profile[mask_high] = amplitude * np.exp((t0-x[mask_high])/(tau*skew))
    
    return profile

@custom_model
def LogConst1D(x, log_amplitude=0.0):
    """
    Constant model. Like astropy.modeling.models.Const1D, except the 
    amplitude is given as the logarithm.
   
    Parameters
    ----------
    x : iterable
        the independent variable

    log_amplitude : float
        the logarithm of the constant amplitude
    """
    return np.ones_like(x) * np.exp(log_amplitude)




def infer_parameters(x, counts, namestr, nwalkers=100, burnin=1000, niter=100):
    # set up the model
    mm = mm = BurstShape() + LogConst1D()
    
    # set up the parameter guesses
    t0_guess = x[np.argsort(counts)[::-1]][0]
    log_amp_guess = np.log(np.mean(np.sort(counts)[::-1][:10]))
    bkg_amp_guess = np.log(np.mean(np.hstack([counts[:10], counts[-10:]])))
    
    # here is the set of starting parameters
    start_pars = [t0_guess, log_amp_guess, 0.2, 1.0, bkg_amp_guess]

    # set up the priors
    
    # exponential prior for the amplitude between some fairly wide boundaries
    p_log_amplitude = lambda logamp: scipy.stats.uniform(np.log(0.01), (np.log(10000)-np.log(0.01))).pdf(logamp)

    # t0 can be uniformely anywhere between the start and end of the light curve
    min_x = x[0]
    max_x = x[-1]
    p_t0 = lambda t0: scipy.stats.uniform(min_x, (max_x - min_x)).pdf(t0)

    # tau can be somewhere between a fraction of the time resolution and the duration of the light curve
    min_tau = np.diff(x)[0]/10.0
    max_tau = max_x - min_x
    p_tau = lambda tau: scipy.stats.uniform(min_tau, (max_tau - min_tau)).pdf(tau)

    # skew can be anywhere between -20 and 20:
    min_skew = -20
    max_skew = 20
    p_skew = lambda skew: scipy.stats.uniform(min_skew, (max_skew - min_skew)).pdf(skew)

    # log background amplitude is another exponential distribution same as burst amplitude
    p_log_bkg_amplitude = lambda log_bkg: scipy.stats.uniform(np.log(0.01), (np.log(10000)-np.log(0.01))).pdf(log_bkg)

    # make a dictionary of the type {param_name: prior}
    priors = {"log_amplitude_0": p_log_amplitude,
              "log_amplitude_1": p_log_bkg_amplitude,
              "t0_0": p_t0,
              "tau_0": p_tau,
              "skew_0": p_skew}

    # set up the Posterior object
    lpost = PoissonPosterior(x, counts, mm, priors=priors)

    # set up the class for parameter estimation
    parest = ParameterEstimation(max_post=True)

    # do a model fit for seeding the MCMC chains
    res = parest.fit(lpost, start_pars, neg=True)

    # perform MCMC sampling
    sample_res = parest.sample(lpost, res.p_opt, cov=res.cov, nwalkers=nwalkers, 
                               niter=niter, burnin=burnin, threads=1)

    # plot some diagnostics
    ndim = sample_res.samples.shape[1]

    fig, axes = plt.subplots(ndim, 1, figsize=(8,3*ndim), sharex=True)
    axes = np.hstack(axes)

    param_names = [r"$t_0$", r"$\log(A_{\mathrm{burst}})$", 
                   r"$\tau$", r"$s$", r"$\log(A_{\mathrm{bkg}})$"]

    for i in range(sample_res.samples.shape[1]):
        axes[i].plot(sample_res.samples[:,i])
        axes[i].set_ylabel(param_names[i])

    plt.tight_layout()
    plt.savefig(namestr + "_trace.png", format="png")
    plt.close()

    # make corner plot
    fig = corner.corner(sample_res.samples, labels=param_names);
    
    plt.tight_layout()
    plt.savefig(namestr + "_corner.png", format="png")
    plt.close

    # save the outputs to file
    np.savetxt(namestr + "_samples.txt", sample_res.samples)

    return

def main():

    if mode == "single":
        data = np.loadtxt(filename)
        x = data[:,0]
        counts = data[:,1]

        namestr = "".join(filename.split(".")[:-1])

        infer_parameters(x, counts, namestr, nwalkers=nwalkers, burnin=burnin, niter=niter)
   
    else:
        files = np.loadtxt(filename, dtype=str)

        for f in files:
             data = np.loadtxt(f)
             x = data[:,0]
             counts = data[:,1]

             namestr =  "".join(f.split(".")[:-1])
             infer_parameters(x, counts, namestr, nwalkers=nwalkers, burnin=burnin, niter=niter)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script to do parameter estimation for all bursts. Data must be in text files, two columns of type (time stamps, counts).")

    modechoice = parser.add_mutually_exclusive_group(required=True)
    modechoice.add_argument('--single', action='store_true', dest='single', help='run on single burst')
    modechoice.add_argument('--multiple', action='store_true', dest='multiple', help='run on multiple bursts at once')

    # general command line arguments
    parser.add_argument("-w", "--nwalkers", action="store", type=int, dest="nwalkers", required=False, default=100, help="The number of walkers in the MCMC routine.")
    parser.add_argument("-i", "--niter", action="store", type=int, dest="niter", required=False, default=100, help="The number of iterations to run each chain.")
    parser.add_argument("-b", "--burnin", action="store", type=int, dest="burnin", required=False, default=1000, help="The number of steps for burning in the chains.")


    parser.add_argument('-f', '--filename', action="store", dest='filename', help='Filename of either the data file (single mode) or of a file with a list of files to be read and analysed (multiple mode). List needs to include full path to data files!')

    # parse command line arguments
    clargs = parser.parse_args()

    # set mode depending on whether we're looking at one burst or multiple
    if clargs.single == True and clargs.multiple == False:
        mode = 'single'

    elif clargs.multiple == True and clargs.single == False:
        mode = 'multiple'

    # get the parameters out into useful variables
    burnin = clargs.burnin
    niter = clargs.niter
    nwalkers = clargs.nwalkers

    filename = clargs.filename

    # run main() function
    main()


