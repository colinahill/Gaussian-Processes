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

def line(m,c,x):
    return m*x + c

def sine(amp,period,x):
    return amp*np.sin(2.0*np.pi*x/period)

def gen_data(p):
    # Reproducible results!
    np.random.seed(123)
    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(100*np.random.rand(N))
    yerr = 0.1+0.5*np.random.rand(N)
    y = line(p[0],p[1],x) + sine(p[2],p[3], x) + yerr-0.35
    # y = line(p[0],p[1],x) + yerr-0.35
    return x, y, yerr

def model(params, x):
    m, c = params
    return m*x + c

def plotposterior(samples, data, outfile):
    fig, ax = pl.subplots()

    # Plot the data.
    ax.errorbar(data[0],data[1], yerr=data[2], fmt=".k", capsize=0)

    # The positions where the prediction should be computed.
    x = np.linspace(data[0][0],data[0][-1],100)

    nposterior = 24
    # Plot nposterior samples.
    for s in samples[np.random.randint(len(samples), size=nposterior)]:
        ax.plot(x, model(s, x), color="#4682b4", alpha=0.3)
    fig.savefig(outfile, dpi=150)

def results(samples, truth):
    amp, period = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.85, 50, 84.15], axis=0)))

    print("""MCMC result:
        amp    = {0[0]} +{0[1]} -{0[2]} (truth: {1})
        period = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    """.format(amp, truth[0], period, truth[1]))

def resultsgp(samples):
    amp, period, a, tau = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.85, 50, 84.15], axis=0)))

    print("""MCMC result:
        amp    = {0[0]} +{0[1]} -{0[2]}
        period = {1[0]} +{1[1]} -{1[2]}
        a      = {2[0]} +{2[1]} -{2[2]}
        tau    = {3[0]} +{3[1]} -{3[2]}
    """.format(amp, period, a, tau))



################################################################################################
####### These are for independent errors #######################################################
################################################################################################

###### Log likelihood of model, given the data 
def lnlike(p, t, y, yerr):
    return -0.5 * np.sum(((y - model(p, t))/yerr) ** 2)

###### Log of Prior - assumed to be uniform (no preferred values of each parameter)
def lnprior(p):
    m, c = p
    if (0.1 < m < 1.0 and 1 < c < 10.0):
        return 0.0
    return -np.inf

###### Log probability of Prior
def lnprob(p, x, y, yerr):
    lp = lnprior(p)
    return lp + lnlike(p, x, y, yerr) if np.isfinite(lp) else -np.inf

def fit_ind(initial, data, nwalkers, nburn, nrun):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    print("Running production...")
    p0, _, _ = sampler.run_mcmc(p0, nrun)

    return sampler

################################################################################################
####### Gaussian process functions       #######################################################
################################################################################################

##### Log likelihood of model, given the data 
def lnlike_gp(p, t, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.CosineKernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model(p[2:], t))

###### Log of Prior - assumed to be uniform (no preferred values of each parameter)
def lnprior_gp(p):
    m, c,a, tau = p
    if (0.5 < a < 1.5 and  20 < tau < 40 and 0.1 < m < 1.0 and 1 < c < 10.0):
        return 0.0
    return -np.inf

###### Log probability of Prior
def lnprob_gp(p, x, y, yerr):
    lp = lnprior_gp(p)
    return lp + lnlike_gp(p, x, y, yerr) if np.isfinite(lp) else -np.inf

def fit_gp(initial, data, nwalkers, nburn, nrun):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, nrun)

    return sampler

################################################################################################

###########################################
###### Generate data
#####################################################
### m, c, amp, period
p = [0.2, 5.0, 1.0, 30.0]
x,y,yerr = gen_data(p)
data = (x,y,yerr)

# autotime = emcee.autocorr.integrated_time(x, low=0, high=None, step=1, c=1, full_output=False, axis=0, fast=False)
# print("Autotime = ", autotime)


### True parameters
truth = p[:2]
initial = p[:2]
labels = ['m','c']

####### Do the least-squares fit and compute the uncertainties.
# A = np.vstack((np.ones_like(x), x)).T
# C = np.diag(yerr * yerr)
# cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
# c_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
# print("""Least-squares results:
#     m = {0} ± {1} (truth: {2})
#     c = {3} ± {4} (truth: {5})
# """.format(m_ls, np.sqrt(cov[1, 1]), truth[0], c_ls, np.sqrt(cov[0, 0]), truth[1]))

# pl.errorbar(x,y,yerr=yerr,fmt='o')
# xi = np.linspace(0,100,500)
# pl.plot(xi, line(m_ls, c_ls, xi), "--k")
# pl.show()

# ################################################################################################

nburn = 200
nrun = 2000
nwalkers = 200


##### Independent errors
# print("Fitting independent")
# sampler = fit_ind(initial, data, nwalkers, nburn, nrun)
# print("Making plots")
# samples = sampler.flatchain
# fig = corner.corner(samples, truths=truth, labels=labels)
# fig.savefig("ind-corner.png", dpi=150)
# plotposterior(samples,data,"ind-posterior.png")
# results(samples, truth)


# fig, ax = pl.subplots(2,sharex=True)
# ###### amp, period, phi
# for i in range(nwalkers):
#     ax[0].plot(sampler.chain[i,:,0],'black')
#     ax[1].plot(sampler.chain[i,:,1],'black')
# ax[0].set_ylabel("m")
# ax[0].axhline(y=truth[0],color='blue',ls='-',lw=2)
# ax[1].set_ylabel("c")
# ax[1].axhline(y=truth[1],color='blue',ls='-',lw=2)
# ax[1].set_xlabel("Steps")

# pl.show()

####### For GP fitting
initial_gp = p
print("Fitting GP")
sampler = fit_gp(initial_gp, data, nwalkers, nburn, nrun)
print("Making plots")
samples = sampler.flatchain
# fig = corner.corner(samples[:, 2:], truths=truth, labels=labels)
# fig.savefig("gp-corner.png", dpi=150)
# plotposterior(samples[:, 2:],data,"gp-posterior.png")
# results(samples[:,2:], truth)
fig = corner.corner(samples, truths=p, labels=['m','c','a','tau'])
fig.savefig("gp-corner.png", dpi=150)
# plotposterior(samples,data,"gp-posterior.png")
resultsgp(samples)

fig, ax = pl.subplots(4,sharex=True)
###### m,c,a,tau
for i in range(nwalkers):
    ax[0].plot(sampler.chain[i,:,0],'black')
    ax[1].plot(sampler.chain[i,:,1],'black')
    ax[2].plot(sampler.chain[i,:,2],'black')
    ax[3].plot(sampler.chain[i,:,3],'black')
ax[0].set_ylabel("m")
ax[0].axhline(y=p[0],color='blue',ls='-',lw=2)
ax[1].set_ylabel("c")
ax[1].axhline(y=p[1],color='blue',ls='-',lw=2)
ax[2].set_ylabel("a")
ax[2].axhline(y=p[2],color='blue',ls='-',lw=2)
ax[3].set_ylabel("tau")
ax[3].axhline(y=p[3],color='blue',ls='-',lw=2)
ax[3].set_xlabel("Steps")

pl.show()
