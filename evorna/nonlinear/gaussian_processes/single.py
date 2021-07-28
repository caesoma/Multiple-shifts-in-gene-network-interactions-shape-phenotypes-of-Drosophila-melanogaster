#!/usr/bin/python3

"""
Functions for all kinds of Gaussian Process calculations (including those with posteriors)
"""

import numpy
import numba
import timeit

import evorna.nonlinear.bayesian as bayesian

@numba.jit
def se_kernel_pt(xp, xq, ella, ellb):
    """squared exponential kernel for a pair of points:
    xq and xp are the two points, ella and ellb are the bandwidths associated to the data from each data points
    (i.e. the function admits the general case with multiple channels) """

    corr = numpy.exp((-1/((ella**2)+(ellb**2)))*numpy.abs(xp-xq)**2)

    return corr

# functions to compute the 'K' matrices of the gaussian processes

def single_k_matrix(x, lparam, sigmap2):
    """ computes the coavariance matrix for training points of a single task, a single bandwidth, and variance parameter """

    N = len(x)
    K = numpy.full([N, N], numpy.NaN);

    la,lb = lparam, lparam
    for i in range(N):
        for j in range(N):
            K[i, j] = sigmap2*se_kernel_pt(x[i], x[j], la, lb)

    return K


def single_ks_matrix(x, xPredicted, lparam, sigmap2):
    """ computes the coavariance matrix between training and test points of a single task, i.e. single bandwidth, and variance parameter """

    N = len(x)
    testN = len(xPredicted)

    Ks = numpy.full([testN, N], -numpy.Infinity)

    la,lb = lparam, lparam
    for i in range(testN):
        for j in range(N):
            Ks[i,j] = sigmap2*se_kernel_pt(xPredicted[i], x[j], la, lb)

    return Ks


def single_cholesky_L(x, params):
    """ Computes K matrix and its Cholesky decomposition for a single-channel Gaussian Process """

    mu, kfDiag, ells, sigman = params
    N = len(x)

    Kn = single_k_matrix( x, ells, kfDiag )

    sigman2 = numpy.repeat(sigman, N)**2
    L = numpy.linalg.cholesky( Kn + numpy.diag(sigman2) )

    return L, Kn


def singlegp_latent_f(x, params, f_til):
    """ Computes GP vector for a single channel given standard normal independent draws (`f_til`) """

    mu, kfDiag, ells, sigman = params

    L, Kn = single_cholesky_L(x, params)

    latentf = numpy.dot( L, f_til )

    return latentf, L, Kn


def single_latent_gp(xPredicted, xSampled, params, f1_til, f2_til):
    """ Computes intermediate (predicted/interpolated) points for two replicate draws from a single-channel Gaussian Process """

    mu, kfDiag, ells, sigman = params
    N = len(xSampled)
    testN = len(xPredicted)

    muVec = numpy.repeat(mu, N)

    L, Kn = single_cholesky_L( xSampled, params )

    latentf1 = numpy.dot( L, f1_til )
    latentf2 = numpy.dot( L, f2_til )

    alpha1 = numpy.linalg.solve( numpy.transpose( L ), numpy.linalg.solve( L, latentf1 ) )
    alpha2 = numpy.linalg.solve( numpy.transpose( L ), numpy.linalg.solve( L, latentf2 ) )

    Ks = single_ks_matrix( xSampled, xPredicted, ells, kfDiag)

    gpost1 = numpy.full( testN, numpy.NaN )
    gpost2 = numpy.full( testN, numpy.NaN )

    gpost1 = numpy.dot( Ks , alpha1 )
    gpost2 = numpy.dot( Ks , alpha2 )

    return ( ( mu + gpost1 ), ( mu + gpost2 ), (muVec + latentf1), (muVec + latentf2) )


def single_gp_posterior(xPredicted, xSampled, dictraces, mu=0, burnin=-1, verbo=False):
    """ Computes Gaussian Process output function for two replicate draws from a single-channel Gaussian Process """

    S, L = dictraces['KfDiag'].shape

    N = len(xSampled)
    interN = len(xPredicted)

    # load posteriors for model variables
    kfDiagArray = dictraces['KfDiag']
    ellArray = dictraces['ell']
    f1tilArray = dictraces['f1_til']
    f2tilArray = dictraces['f2_til']
    # alphaArray =  dictraces['alpha']

    # noise parameter is not estimated for non-gaussian observations, but a negligible value is introduced for numerical stability
    sigman = 1e-4

    # chose samples number of samples to discard
    burnindex = int(burnin) if ( burnin >= 0 ) else int( dictraces['alpha'].shape[0] / 2 )

    print(">>> Thinned burn-in:", burnindex) if (verbo==True) else None
    print(">>> Thinned posterior samples:", (S-burnindex)) if (verbo==True) else None

    gparray1 = numpy.full( [ (S-burnindex), L, interN ], numpy.NaN )
    gparray2 = numpy.full( [ (S-burnindex), L, interN ], numpy.NaN )

    mulatent1 = numpy.full( [ (S-burnindex), L, N ], numpy.NaN )
    mulatent2 = numpy.full( [ (S-burnindex), L, N ], numpy.NaN )

    for l in range(L):
        print(">>> chain:", l) if (verbo==True) else None
        print(">>> sample:", end=" ") if (verbo==True) else None
        for bni,i in enumerate( range( burnindex, S ) ):
            if ( ( (i % ( int(S) / 100 ) ) == 0 ) and (verbo==True) ):
                print(i)  #, end=", ")

            params = mu, kfDiagArray[i,l], ellArray[i,l], sigman
            f1_til = f1tilArray[i,l]
            f2_til = f2tilArray[i,l]

            try:
                gparray1[bni,l,:], gparray2[bni,l,:], mulatent1[bni,l,:], mulatent2[bni,l,:] = single_latent_gp(xPredicted, xSampled, params, f1_til, f2_til)
            except:
                warnstring = "error when computing gp at iteration: " + str(bni)
                warnings.warn(warnstring, UserWarning)


    return gparray1, gparray2, mulatent1, mulatent2


def evorna_single_gp_posterior(resultspath, cmdstanDict, fbid, subdata, metadata, posterior=[], verbo=False):
    """ Computes Gaussian Process output function for two replicate draws and all selection schemes for a single-channel Gaussian Process """

    tic = timeit.default_timer()
    if ( len(posterior) == -1 ): # this option is disabled because the `load_cmdstan_singlegp` function is absent from the `bayesian` module, but see multigp equivalent in case this is needed
        print(">>> Loading trace files\n>>> computing GP output ") if (verbo==True) else None
        # if posteriors are not provided as argument the function will try to load traces from file, which must follow naming template: <modelname>_<selscheme>_<fbid>_nuts_<iterations>_chain<l>.csv
        dictraceShort, samplaramsShort = load_cmdstan_singlegp(resultspath, cmdstanDict['modelname'] + "_short_" + fbid + "_nuts" + str(cmdstanDict['iterations']) + "_chain", cmdstanDict, N, "short", muvar=False)

        dictraceControl, samplaramsControl = load_cmdstan_singlegp(resultspath, cmdstanDict['modelname'] + "_control_" + fbid + "_nuts" + str(cmdstanDict['iterations']) + "_chain", cmdstanDict, N, "control", muvar=False)

        dictraceLong, samplaramsLong = load_cmdstan_singlegp(resultspath, cmdstanDict['modelname'] + "_long_" + fbid + "_nuts" + str(cmdstanDict['iterations']) + "_chain", cmdstanDict, N, "long", muvar=False)

    else:
        print(">>> Computing GPs from provided posteriors ") if (verbo==True) else None
        dictraceShort, dictraceControl, dictraceLong = posterior

    # get generations with observed data and where to plot the GP function
    generations = metadata[(metadata.sel=="short") & (metadata.Rep==1) & (metadata.pool==1)].generation.values
    continuousGen = numpy.arange(generations[0], generations[-1]+1.1, 0.1)

    shortMu = numpy.log( numpy.mean( subdata[metadata.sel=="short"]) )
    controlMu = numpy.log( numpy.mean( subdata[metadata.sel=="control"]) )
    longMu = numpy.log( numpy.mean( subdata[metadata.sel=="long"]) )

    burnthin = int( cmdstanDict['burnin']/cmdstanDict['thinning'] ) if (cmdstanDict['washburn']==1) else 0
    print(">>> Burning initial samples:", burnthin) if (verbo==True) else None


    gPosterior1s, gPosterior2s, mulatent1s, mulatent2s = single_gp_posterior(continuousGen, generations, dictraceShort, mu=shortMu, burnin=burnthin, verbo=verbo )

    gPosterior1c, gPosterior2c, mulatent1c, mulatent2c = single_gp_posterior(continuousGen, generations, dictraceControl, mu=controlMu, burnin=burnthin, verbo=verbo  )

    gPosterior1l, gPosterior2l, mulatent1l, mulatent2l = single_gp_posterior(continuousGen, generations, dictraceLong, mu=longMu, burnin=burnthin, verbo=verbo  )

    gPosterior={'shortRep1':gPosterior1s,
                'shortRep2':gPosterior2s,
                'controlRep1':gPosterior1c,
                'controlRep2':gPosterior2c,
                'longRep1':gPosterior1l,
                'longRep2':gPosterior2l }

    tac = timeit.default_timer()
    print(">>> summary computation time:", (tac-tic)/60, "minutes") if (verbo==True) else None

    return gPosterior
