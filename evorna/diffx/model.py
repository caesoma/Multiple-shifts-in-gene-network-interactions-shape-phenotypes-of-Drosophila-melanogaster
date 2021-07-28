#/usr/bin/python3

import numpy
import pandas
import scipy.stats

def negative_binomial_loglikelihood(mean, dispersion, observed):
    """ computes negative binomial from mean (mu) and dispersion (alpha) parameters, by first computing the probability of success `p` and computing the likelihood from a formulation with `p` and `alpha`"""
    mu = mean
    alpha = dispersion

    p = alpha / (mu + alpha)

    logp = numpy.sum( numpy.log( scipy.stats.nbinom( alpha, p ).pmf( observed ) ) )

    return logp

def likelihood_ratio_test(logpFull, logpReduced, df):
    """computes the chi-squared distribution p-value for the likelihood ratio given the negative log likelihoods and the number of degrees of freedom"""

    logpRatio = 2*(logpFull-logpReduced)  # -2LogP_reduced-(-2LogP_full)
    pvalue = 1-scipy.stats.distributions.chi2.cdf(logpRatio, df)

    return pvalue, logpRatio


def linear(coefficients, designMatrix):
    """ linear model as matrix operation, e.g. ["control1", "control2", "short1", "short2", "long1", "long2", "generation", "genXshort1", "genXshort2", "genXlong1", "genXlong2"] ->   y = mu + Bs*'short' + Bl*'long' + Bg*generation + Bgs*generation*'short' + Bgl*generation*'long' """

    coefs = numpy.array(coefficients)

    logqj = numpy.dot(designMatrix, coefs)

    return logqj
