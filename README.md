# A collection of codes that use Gaussian process regression to model a function with correlated noise
All codes require Python, Numpy, Scipy, emcee

The sinusoid model can be used to detect planets around stars using the radial velocity method, as stellar activity can be treated as correlated noise and incorporated into the GP through the covariance kernel, incorporating several hyperparameters that can be marginalized over to obtain the stellar parameters as well as as recover the planetary signal.
