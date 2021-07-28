#!/usr/bin/python3

import numpy

def hat_var_plus_W(θ, quiet=True):
    """ Computes hatVarPlus ('^var+') from between-chain variance-related metric (B) and within-chain variance (W) and returns the the first and last of the three"""

    if len(θ.shape)==2:
        N = θ.shape[0]
        L = θ.shape[1]
    else:
        raise ValueError(">>> This array must have two dimensions, the first being the samples drawn from a chain, the second the parallel chains frow which these were drawn")

    # check that number of chains is smaller than number of samples, which is most likely the case
    if L>N:
        if (not quiet):
            warnstring = ">>> WARNING: number of chains is larger than number of samples per chain, make sure row dimension are different chains, and column dimension are samples from the same chain"
            warnings.warn(warnstring, UserWarning)
    else:
        if (not quiet):
            print(">>> number of chains:", L)
            print(">>> number of samples per chain:", N)

    θ_dot = numpy.mean(θ, axis=0) # compute mean chain value (i.e. mean over all samples of each separate chain)
    θ_dotdot = numpy.mean(θ) # compute overall mean of array (per variable)

    sqrdiff = lambda θ_dotj: ( θ_dotj - θ_dotdot )**2  # f(x) = (x-μ)^2
    B = ( N / (L-1) ) * numpy.sum( list( map( sqrdiff, θ_dot) ) )


    # compute chain variance value (similar to mean, above)
    s2j = numpy.var(θ, axis=0, ddof=1)

    # compute measure of within-chain variance
    W = numpy.mean(s2j)

    # compute aggregated variance metric and divide by within-chain measure to get final metric
    hatVarPlus = ((N-1)/N)*W + B/N

    return hatVarPlus, W


def hat_R(θ):
    """ Computes R^ ('R hat') metric for within vs between MCMC chain variation from an array of N samples x L chains"""
    hatVarPlus, W = hat_var_plus_W(θ)

    hatR = numpy.sqrt(hatVarPlus/W)

    return hatR


def autocorrelation(θ):
    """ computes the autocorrelation at all possible lags in the markov chain """
    N = θ.shape[0]
    L = θ.shape[1]

    hatVarPlus, W = hat_var_plus_W(θ)

    V = lambda t: (1/(L*(N-t))) * numpy.sum(numpy.power(θ[t:N,:] - θ[0:(N-t),:], 2))  # Vt_ij = 1/L(N-t) Σi Σj (θ_i,j - θ_i-t,j)^2
    variogram = list( map( V, range(0,N) ) )

    ρ = 1 - ( variogram / ( 2 * hatVarPlus ) )

    return ρ


def multi_autocorr(θ):
    """ computes the autocorrelation at all possible lags in the markov chain """
    N = θ.shape[0]
    L = θ.shape[1]

    Vl = lambda t: ( 1 / ( N - t ) ) * numpy.sum( numpy.power( θ[t:N,:] - θ[0:(N-t),:], 2 ), axis=0 )  # Vt_ij = 1/L(N-t) Σi Σj (θ_i,j - θ_i-t,j)^2
    variogram = list( map( Vl, range(0,N) ) )

    ρ = 1 - ( variogram / numpy.var(θ, axis=0) )

    return ρ


def f_maxt(t,N,ρ):
    if ( t < N-2 ) and ( ( ρ[t+1] + ρ[t+2] ) >= 0 ):
        f_maxt( t+2, N, ρ )
    else:
        T = t
    return T


def effective_sample_size(θ):
    """ computes the effective number of samples given an array with parallel Markov Chains """
    N = θ.shape[0]
    L = θ.shape[1]

    #hatVarPlus, W = hatVarPlus(θ)
    ρ = autocorrelation(θ)

    maxt = lambda t,N,ρ: maxt( t+2, N, ρ ) if ( ( t < N-2 ) and ( ( ρ[t+1] + ρ[t+2] ) >= 0 ) ) else t  # see f_maxt for clear def

    t=1
    T = maxt( t, N, ρ )

    hatESS = ( N * L ) / ( 1 + ( 2 * ( numpy.sum( ρ[1:T] ) ) ) )

    return hatESS


def vector_ess(tθ):
    """ computes the effective number of samples given an array with parallel Markov Chains """
    V = tθ.shape[2] if len(tθ.shape)==3 else 1

    essi = lambda i: effective_sample_size(tθ[:,:,i]) if len(tθ.shape)==3 else effective_sample_size(tθ)

    vess = list( map( essi, range(V) ) )

    return numpy.array(vess)


def vector_autocorrelation(tθ):
    """ computes the effective number of samples given an array with parallel Markov Chains """
    V = tθ.shape[2] if len(tθ.shape)==3 else 1

    aci = lambda i: autocorrelation(tθ[:,:,i]) if len(tθ.shape)==3 else autocorrelation(tθ)
    #  if len(tθ.shape)==3:
    #     aci = lambda i: autocorrelation(tθ[:,:,i])
    # else:
    #     aci = autocorrelation(tθ)

    vac = list( map( aci, range(V) ) )

    return numpy.transpose(vac)


def vector_hat_R(tθ):
    """ computes the effective number of samples given an array with parallel Markov Chains """
    V = tθ.shape[2] if len(tθ.shape)==3 else 1

    hatRi = lambda i:hat_R(tθ[:,:,i]) if len(tθ.shape)==3 else hat_R(tθ)

    vhatR = list( map( hatRi, range(V) ) )

    return numpy.array(vhatR)


"""
# tests and examples
import numpy
from numpy.random import normal as rgaussian

import matplotlib.pyplot as pyplot
from matplotlib.pyplot import figure, plot, show

varn = 3
alice = 4
chainLength = 10000

randomWalks = numpy.full([chainLength, alice, varn], numpy.NaN)
randomWalks[0,:,0] = 500
randomWalks[0,:,1] = 800
randomWalks[0,:,2] = 1000

nonRandomWalks = numpy.full([chainLength, alice, varn], numpy.NaN)
nonRandomWalks[0,:,0] = 500
nonRandomWalks[0,:,1] = 800
nonRandomWalks[0,:,2] = 1000

for i in range(1,chainLength):
    randomWalks[i,:,:] = randomWalks[i-1,:,:] + rgaussian(0,1,size=(alice, varn))

for i in range(1,chainLength):
    nonRandomWalks[i,:,:] = nonRandomWalks[0,:,:] + rgaussian(0,1,size=(alice,varn))

figure(); [plot(10**numpy.log10(randomWalks[:,:,var])) for var in range(varn)];
figure(); [plot(10**numpy.log10(nonRandomWalks[:,:,var])) for var in range(varn)]; show()

thinning = 10
thinWalks = numpy.full([int(chainLength/thinning), alice, varn], numpy.NaN)

for i in range(int(chainLength/thinning)):
     thinWalks[i,:] = randomWalks[i*thinning,:]


range(int(chainLength/thinning)),
figure(); [plot(10**numpy.log10(randomWalks[:,:,var])) for var in range(varn)];
[plot(10**numpy.log10(thinWalks[:,:,var])) for var in range(varn)];
show()
"""
