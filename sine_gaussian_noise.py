#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import emcee
import corner
import numpy as np
import george
from george import kernels
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

################################################################################################
####### General functions
################################################################################################

def model(p, x):
    amp, period = p
    return amp * np.sin(2*np.pi*x/period)

def generate_data(p, N, rng):
    amp, period, a, b = p
    # Generate some synthetic data from the model.
    gp = george.GP(a * kernels.ExpKernel(b))
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
    y = gp.sample(t)
    y += model(p[:2], t)
    yerr = 0.2 * np.random.rand(N)
    # yerr = 0.1*amp + 0.05 * np.random.rand(N)
    # y += yerr * np.random.randn(N)
    return t, y, yerr

################################################################################################
####### These are for independent errors #######################################################
################################################################################################

###### Log of Prior - assumed to be uniform (no preferred values of each parameter)
def lnprior_ind(p):
    amp, period = p
    if (0.1 < amp < 2 and 0.1 < period < 10):
        return 0.0
    return -np.inf

###### Log likelihood of model, given the data 
def lnlike_ind(p, t, y, invar):
    return -0.5 * np.sum((y - model(p,t)) ** 2 * invar)

###### Log probability of Prior
def lnprob_ind(p, t, y, invar):
    lp = lnprior_ind(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_ind(p, t, y, invar)

def fit_ind(initial, data, nwalkers, nburn, nrun):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ind, args=data)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, nrun)

    return sampler

def results(samples, truth):
    amp, period = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.85, 50, 84.15], axis=0)))

    print("""Independent MCMC result:
        amp    = {0[0]} +{0[1]} -{0[2]} (truth: {1})
        period = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    """.format(amp, truth[0], period, truth[1]))



################################################################################################
####### Gaussian process functions       #######################################################
################################################################################################

###### Log of Prior - assumed to be uniform (no preferred values of each parameter)
def lnprior_gp(p):
    amp, period, a, b = p
    if (0.1 < amp < 2 and 0.1 < period < 10 and 0.1 < a < 0.4 and 1 < b < 6):
        return 0.0
    return -np.inf

###### Log likelihood of model, given the data 
def lnlike_gp(p, t, y, yerr):
    a = p[2]
    b = np.exp(p[3])
    gp = george.GP(a*kernels.ExpKernel(b))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model(p[:2], t))

###### Log probability of Prior
def lnprob_gp(p, x, y, yerr):
    lp = lnprior_gp(p)
    return lp + lnlike_gp(p, x, y, yerr) if np.isfinite(lp) else -np.inf

def fit_gp(initial, data, nwalkers, nburn, nrun):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, nrun)

    return sampler

def results_gp(samples, truth):
    amp, period, a, th = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.85, 50, 84.15], axis=0)))

    print("""GP MCMC result:
        amp    = {0[0]} +{0[1]} -{0[2]} (truth: {1})
        period = {2[0]} +{2[1]} -{2[2]} (truth: {3})
        a      = {4[0]} +{4[1]} -{4[2]} (truth: {5})
        b     = {6[0]} +{6[1]} -{6[2]} (truth: {7})
    """.format(amp, truth[0], period, truth[1], a , truth[2], b, truth[3]))

###############################################################
###############################################################

if __name__ == "__main__":
    np.random.seed(1234)

    truth = [1, 3]
    truth_gp = truth + [0.1, 5]

    low = 0.0
    high = 10.0
    rng = (low,high)
    x = np.linspace(low,high,1000)

    nburn = 2000
    nrun = 20000
    nwalkers = 200

    t, y, yerr = generate_data(truth_gp, 50, rng)
    pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    pl.plot(x,model(truth,x),'k-')
    pl.ylabel(r"$y$")
    pl.xlabel(r"$t$")
    pl.xlim(rng)
    pl.title("simulated data")
    pl.savefig("data.png", dpi=150)



    # Fit assuming independent errors.
    print("Fitting independent")
    data = (t, y, 1.0 / yerr ** 2)
    initial_ind = [i*0.9 for i in truth]
    sampler = fit_ind(initial_ind, data, nwalkers, nburn, nrun)
    # Plot the samples in data space.
    print("Making plots")
    samples = sampler.flatchain
    for s in samples[np.random.randint(len(samples), size=24)]:
        pl.plot(x, model(s, x), color="#4682b4", alpha=0.3)
    pl.title("results assuming uncorrelated noise")
    pl.savefig("ind-results.png", dpi=150)
    labels = ["amp", "period"]
    fig = corner.corner(samples, truths=truth, labels=labels)
    fig.savefig("ind-corner.png", dpi=150)

    # results(samples,truth)
    # samples = sampler.chain[:, 50:, :].reshape((-1, 2))
    # pl.subplot(2, 1, 1)
    # pl.plot(samples[:,0])
    # pl.ylabel(labels[0])
    # pl.subplot(2, 1, 2)
    # pl.plot(samples[:,1])
    # pl.ylabel(labels[1])
    # pl.show()

    # # Fit assuming GP.
    # print("Fitting GP")
    # data = (t, y, yerr)
    # # initial_gp = truth_gp
    # initial_gp = [i*0.8 for i in truth_gp]
    # sampler = fit_gp(initial_gp, data, nwalkers, nburn, nrun)
    # # Plot the samples in data space.
    # print("Making plots")
    # samples = sampler.flatchain
    # pl.figure()
    # pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    # for s in samples[np.random.randint(len(samples), size=24)]:
    #     gp = george.GP(s[2]*kernels.ExpSquaredKernel(s[3]))
    #     gp.compute(t, yerr)
    #     m = gp.sample_conditional(y - model(s[:2], t), x) + model(s[:2], x)
    #     pl.plot(x, m, color="#4682b4", alpha=0.3)
    # pl.ylabel(r"$y$")
    # pl.xlabel(r"$t$")
    # pl.xlim(low,high)
    # pl.plot(x,model(truth,x),'k-')
    # pl.title("results with Gaussian process noise model")
    # pl.savefig("gp-results.png", dpi=150)
    # # # Make the corner plot.
    # labels = ["amp", "period","a","th"]
    # fig = corner.corner(samples, truths=truth_gp, labels=labels)
    # fig.savefig("gp-corner.png", dpi=150)
    # results_gp(samples,truth_gp)



# #     # pl.close('all')

# # #     fig, ax = pl.subplots(4,sharex=True)
# # # ###### amp, period, phi
# # #     for i in range(nwalkers):
# # #         ax[0].plot(sampler.chain[i,:,0],'black')
# # #         ax[1].plot(sampler.chain[i,:,1],'black')
# # #         ax[2].plot(sampler.chain[i,:,2],'black')
# # #         ax[3].plot(sampler.chain[i,:,3],'black')
# # #     ax[0].set_ylabel("Amplitude")
# # #     ax[0].axhline(y=truth_gp[0],color='blue',ls='-',lw=2)
# # #     ax[1].set_ylabel("Period)")
# # #     ax[1].axhline(y=truth_gp[1],color='blue',ls='-',lw=2)
# # #     ax[2].set_ylabel("a")
# # #     ax[2].axhline(y=truth_gp[2],color='blue',ls='-',lw=2)
# # #     ax[3].set_ylabel("th")
# # #     ax[3].axhline(y=truth_gp[3],color='blue',ls='-',lw=2)
# # #     ax[3].set_xlabel("Steps")

# # #     pl.show()
