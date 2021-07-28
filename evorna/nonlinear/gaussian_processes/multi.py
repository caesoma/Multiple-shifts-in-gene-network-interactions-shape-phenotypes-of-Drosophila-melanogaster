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

@numba.jit
def assemble_Kf(kfDiag, kfTril, verbo=False):
    """ function to assemble covariance matrix from vectors for diagonal and off-diagonal terms """

    M = len(kfDiag);
    Kf = numpy.full([M,M], -numpy.Infinity);
    sumi = 0;
    for i in range(M - 1):
        for j in range((i + 1), M):
            Kf[i, j] = kfTril[sumi];
            Kf[j, i] = Kf[i, j];
            sumi = sumi+1;
            print("matrix: ",i,j) if (verbo==True) else None
            print("vector: ", sumi) if (verbo==True) else None

        Kf[i, i] = kfDiag[i];
    Kf[M-1,M-1] = kfDiag[M-1];
    return Kf;

# zero-index function to assemble
@numba.jit
def assemble_Cf(kfDiag, kfTril):
    """ function to assemble correlation matrix from vectors for diagonal and off-diagonal terms (similar to `assemble_Kf`, but normalizes covariances by appropriate variance ). This function is not mean for computation, but visualization, so diagonals are set to zero to improve perception """

    M = len(kfDiag);
    Cf = numpy.full([M,M], -numpy.Infinity);
    sumi = 0;
    for i in range(M - 1):
        for j in range((i + 1), M):
            Cf[i, j] = kfTril[sumi] / numpy.sqrt(kfDiag[i]*kfDiag[j]);
            Cf[j, i] = Cf[i, j];
            sumi = sumi+1;
            print("matrix: ",i,j) if (verbo==True) else None
            print("vector: ", sumi) if (verbo==True) else None

        Cf[i, i] = 0;
    Cf[M-1,M-1] = 0;
    return Cf

def disassemble_Kf(Kf):
    """ Opposite of `assemble_Kf` """

    M = len(numpy.diag(Kf))
    trilM = int((M**2 - M)/2)

    kfDiag = numpy.diag(Kf)
    kfTril = numpy.full(trilM, numpy.nan)

    sumi = 0;
    for i in range(M - 1):
        for j in range((i + 1), M):
            kfTril[sumi] = Kf[i, j]
            sumi = sumi+1;

    return kfDiag, kfTril

# functions to compute the 'K' matrices of the gaussian processes
@numba.jit
def multi_k_matrix(x, bandwidths, Kf, N, M, sigman):
    """ computes the covariance matrix for TRAINING points of multiple channels, multiple bandwidths, and a positive semi-definite matrix with process and "cross-process" variance """

    #ylen = M*N
    K = numpy.full([M*N, M*N], numpy.NaN);

    for l in range(0,M):
        ella = bandwidths[l]
        subl = l*N
        for i in range(0,N):
            K[subl+i,subl+i] = Kf[l,l]
            for j in range((i+1),N):
                K[subl+i,subl+j] = Kf[l,l]*se_kernel_pt(x[i], x[j], ella, ella)
                K[subl+j,subl+i] = K[subl+i,subl+j]

        for k in range(0,M):
            ellb = bandwidths[k]
            subk = k*N

            for i in range(0,N):
                K[subl+i,subk+i] = Kf[l,k]
                K[subk+i,subl+i] = Kf[l,k]
                for j in range((i+1),N):
                    K[subl+i,subk+j] = Kf[l][k]*se_kernel_pt(x[i], x[j], ella, ellb)
                    K[subl+j,subk+i] = K[subl+i,subk+j]
                    K[subk+j,subl+i] = K[subl+i,subk+j]
                    K[subk+i,subl+j] = K[subk+j,subl+i]

    sigman2 = numpy.repeat(sigman, N)**2
    Kn = K + numpy.diag(sigman2)  # sigman*numpy.eye(M*N)

    return Kn


@numba.jit
def multi_k_matrix2(x, bandwidths, Kf, N, M, sigman):
    """ computes the covariance matrix for TRAINING points of multiple channels, multiple bandwidths, and  a positive semi-definite matrix with process and "cross-process" variance """

    #ylen = M*N
    K = numpy.full([M*N, M*N], numpy.NaN);

    for l in range(0,M):
        ella = bandwidths[l];
        subl = l*N;

        for k in range(0,M):
            ellb = bandwidths[k];
            subk = k*N;

            for i in range(0,N):
                for j in range(i, N):
                    K[subl+i][subk+j] = Kf[l][k]*se_kernel_pt(x[i], x[j], ella, ellb);
                    K[subk+j][subl+i] = K[subl+i][subk+j];

    sigman2 = numpy.repeat(sigman, N)**2
    Kn = K + numpy.diag(sigman2)

    return Kn


@numba.jit
def multi_ks_matrix(xSampled, xPredicted, bandwidths, Kf, N, M):
    """ computes the coavariance matrix between training and test points of multiple channels, with the appropriate cross-channel parameters """


    testN = len(xPredicted)
    Ks = numpy.full([M, testN, M*N], -numpy.Infinity)

    for l in range(0,M):
        ellb = bandwidths[l]

        for i in range(testN):
            for trn in range(0,M):
                ella = bandwidths[trn]
                subtrn = trn*N

                for j in range(0,N):
                    Ks[l,i,subtrn+j] = Kf[l,trn]*se_kernel_pt( xPredicted[i], xSampled[j], ella, ellb )

    return Ks


@numba.jit
def cholesky_L(x, params):
    """ Computes K matrix and its Cholesky decomposition for a multi-channel Gaussian Process """

    mu, kfDiag, kfTril, ells, sigman = params
    M = len(ells)
    N = len(x)

    Kf = assemble_Kf( kfDiag, kfTril )
    Kn = multi_k_matrix( x, ells, Kf, N, M, sigman)

    L = numpy.linalg.cholesky( Kn )

    return L, Kn, Kf


@numba.jit
def latent_f(x, params, f_til):
    """ Computes GP vector for a multi channel given standard normal independent draws (`f_til`) """

    mu, kfDiag, kfTril, ells, sigman = params

    L, Kn, Kf = cholesky_L(x, params)

    latentf = numpy.dot( L, f_til )

    return latentf, L, Kn, Kf


@numba.jit
def multi_latent_gp(xPredicted, xSampled, params, f_til):
    """ Computes intermediate (predicted/interpolated) points for one draw from a multi-channel Gaussian Process where the normal draws are standard (`f_til`) """

    mu, kfDiag, kfTril, ells, sigman = params
    M = len(ells)
    N = len(xSampled)

    muVec = numpy.repeat(mu, N)

    latentf, L, Kn, Kf = latent_f(xSampled, params, f_til)

    alpha = numpy.linalg.solve( numpy.transpose( L ), numpy.linalg.solve( L, latentf ) )

    Ks = multi_ks_matrix( xSampled, xPredicted, ells, Kf, N, M )

    gpost = numpy.full( [ len( xPredicted ), M ], numpy.NaN )
    for l in range(M):
        gpost[:,l] = numpy.dot( Ks[l,:,:] , alpha )

    return ( (mu + gpost), (muVec + latentf) )


@numba.jit
def gaussian_process(xPredicted, xSampled, params, y):
    """ Computes intermediate (predicted/interpolated) points for one draw from a multi-channel Gaussian Process where the observations are the actual data (`y`) """

    mu, kfDiag, kfTril, ells, sigman = params
    M = len(ells)
    N = len(xSampled)

    mx = numpy.repeat(mu, N)

    L, Kn, Kf = cholesky_L(xSampled, params)

    alpha = numpy.linalg.solve( numpy.transpose( L ), numpy.linalg.solve( L, y - mx ) )

    Ks = multi_ks_matrix( xSampled, xPredicted, ells, Kf, N, M )

    gpost = numpy.full( [ len( xPredicted ), M ], numpy.NaN )
    for l in range(M):
        gpost[:,l] = numpy.dot( Ks[l,:,:] , alpha )

    return (mu + gpost)


@numba.jit
def latent_gp2(xPredicted, xSampled, params, f1_til, f2_til):
    """ Computes intermediate (predicted/interpolated) points for two replicate draws from a multi-channel Gaussian Process where the normal draws are standard (`f1_til`, `f2_til`) """

    mu, kfDiag, kfTril, ells, sigman = params
    M = len(ells)
    N = len(xSampled)

    muVec = numpy.repeat(mu, N)

    L, Kn, Kf = cholesky_L(xSampled, params)

    latentf1 = numpy.dot( L, f1_til )
    latentf2 = numpy.dot( L, f2_til )

    alpha1 = numpy.linalg.solve( numpy.transpose( L ), numpy.linalg.solve( L, latentf1 ) )
    alpha2 = numpy.linalg.solve( numpy.transpose( L ), numpy.linalg.solve( L, latentf2 ) )

    Ks = multi_ks_matrix( xSampled, xPredicted, ells, Kf, N, M )

    gpost1 = numpy.full( [ len( xPredicted ), M ], numpy.NaN )
    gpost2 = numpy.full( [ len( xPredicted ), M ], numpy.NaN )
    for l in range(M):
        gpost1[:,l] = numpy.dot( Ks[l,:,:] , alpha1 )
        gpost2[:,l] = numpy.dot( Ks[l,:,:] , alpha2 )

    return ( ( mu + gpost1 ), ( mu + gpost2 ), (muVec + latentf1), (muVec + latentf2) )


def print_Kf(M):
    "simple loop function to print matrix entries"
    sumi = 0;
    strlen = len(str(int((M**2 - M)/2)))
    pad = (strlen - len(str(sumi)))*' '
    Kf = numpy.full([M,M], "..");
    for i in range(M):
        print(i*(strlen*'.'+' '), end='')
        for j in range((i + 1), M):
            Kf[i, j] = sumi
            print(pad+str(sumi), end=' ')
            sumi = sumi+1;
            pad = (strlen - len(str(sumi)))*' '
        print("\n")

    return Kf


def gp_posterior(xPredicted, xSampled, dictraces, muM, burnin=-1, verbo=False):
    """ Computes Gaussian Process output function as separate channels (an array instead of vector) for two replicate draws from a multi-channel Gaussian Process """

    S, L, M = dictraces['KfDiag'].shape

    N = len(xSampled)
    testN = len(xPredicted)

    # load posteriors for model variables
    kfDiagArray = dictraces['KfDiag']
    kfTrilArray = dictraces['KfTril']
    ellArray = dictraces['ell']
    f1tilArray = dictraces['f1_til']
    f2tilArray = dictraces['f2_til']
    # alphaArray =  dictraces['alpha']

    # noise parameter is not estimated for non-gaussian observations, but a negligible value is introduced for numerical stability
    sigman = M*[1e-4]

    # chose samples number of samples to discard
    burnindex = int(burnin) if ( burnin >= 0 ) else int( dictraces['alpha'].shape[0] / 2 )

    print(">>> Thinned burn-in:", burnindex) if (verbo==True) else None
    print(">>> Thinned posterior samples:", (S-burnindex)) if (verbo==True) else None

    gparray1 = numpy.full([(S-burnindex), L, testN, M], numpy.NaN)
    gparray2 = numpy.full([(S-burnindex), L, testN, M], numpy.NaN)

    mulatent1 = numpy.full([(S-burnindex), L, N*M], numpy.NaN)
    mulatent2 = numpy.full([(S-burnindex), L, N*M], numpy.NaN)

    for l in range(L):
        if (verbo==True):
            print(">>> chain:", l)
            print(">>> sample:", end=" ")
        for bni,i in enumerate( range( burnindex, S ) ):
            if ( ( (i % ( int(S) / 100 ) ) == 0 ) and (verbo==True) ):
                print(i)  #, end=", ")

            params = muM, kfDiagArray[i,l], kfTrilArray[i,l], ellArray[i,l], sigman
            f1_til = f1tilArray[i,l]
            f2_til = f2tilArray[i,l]

            try:
                gparray1[bni,l,:,:], gparray2[bni,l,:,:], mulatent1[bni,l,:], mulatent2[bni,l,:] = latent_gp2(xPredicted, xSampled, params, f1_til, f2_til)
            except:
                warnstring = "error when computing gp at iteration: " + str(bni)
                warnings.warn(warnstring, UserWarning)

    return gparray1, gparray2, mulatent1, mulatent2


def evorna_gp_posterior(tracespath, cmdstanDict, fbidij, subdata, metadata, posterior=[], verbo=False):
    """ Computes Gaussian Process output function as separate channels for all selection schemes for two replicate draws from a multi-channel Gaussian Process """

    tic = timeit.default_timer()

    fbidi, fbidj = fbidij
    subsubdata = subdata[ [fbidi, fbidj] ]

    # if posteriors are not provided as argument the function will try to load traces from file, which must follow naming template: <modelname>_<selscheme><fbid1>_<fbid2>_nuts_<iterations>_chain<l>.csv
    if ( len(posterior) == 0 ):
        print(">>> Loading trace files\n>>> computing GP output ") if (verbo==True) else None
        try:
            [[dictraceShort, samplaramsShort],[dictraceControl, samplaramsControl],[dictraceLong, samplaramsLong]] = bayesian.evorna_posteriors(tracespath, cmdstanDict, fbidij, N, verbo=False)
        except:
            # if order of FBID labels is reversed, file won't be found and an exception will be raised, this clause catches that and reverses the order of the labels to make the rest work (with the opposite order)
            fbidij2 = fbidj, fbidi
            subsubdata = subdata[ [fbidj, fbidi] ]
            [[dictraceShort, samplaramsShort],[dictraceControl, samplaramsControl],[dictraceLong, samplaramsLong]] = bayesian.evorna_posteriors(tracespath, cmdstanDict, fbidij2, N, verbo=False)
    else:
        print(">>> Computing GPs from provided posteriors ") if (verbo==True) else None
        dictraceShort, dictraceControl, dictraceLong = posterior

    # computes credicility limits and cheks overlap between the selectio schemes and controls
    cfMeanij, cf25ij, cf975ij = bayesian.compute_Cf_ij((dictraceShort, dictraceControl, dictraceLong), fbidij, burn=0, verbo=False)
    stars = bayesian.overlap_cf_ij(cfMeanij, cf25ij, cf975ij)


    # get generations with observed data and where to plot the GP function
    generations = metadata[(metadata.sel=="short") & (metadata.Rep==1) & (metadata.pool==1)].generation.values
    continuousGen = numpy.arange(generations[0], generations[-1]+1.1, 0.1)


    burnthin = int(cmdstanDict['burnin']/cmdstanDict['thinning']) if (cmdstanDict['washburn']==1) else 0
    print(">>> Burning initial samples:", burnthin) if (verbo==True) else None

    if (verbo==True):
        print(fbidi, fbidj)
        print("rho: ", cfMeanij)

    shortMu = list( numpy.log( numpy.mean( subsubdata[metadata.sel=="short"], axis=0 ) ) )
    controlMu = list( numpy.log( numpy.mean( subsubdata[metadata.sel=="control"], axis=0 ) ) )
    longMu = list( numpy.log( numpy.mean( subsubdata[metadata.sel=="long"], axis=0 ) ) )

    # compute postiors from GP output
    gPosterior1s, gPosterior2s, mulatent1s, mulatent2s = gp_posterior(continuousGen, generations, dictraces=dictraceShort, muM=shortMu, burnin=burnthin, verbo=verbo)

    gPosterior1c, gPosterior2c, mulatent1c, mulatent2c = gp_posterior(continuousGen, generations, dictraces=dictraceControl, muM=controlMu, burnin=burnthin, verbo=verbo )

    gPosterior1l, gPosterior2l, mulatent1l, mulatent2l = gp_posterior(continuousGen, generations, dictraces=dictraceLong, muM=longMu, burnin=burnthin, verbo=verbo )

    gPosterior={'shortRep1':gPosterior1s,
                'shortRep2':gPosterior2s,
                'controlRep1':gPosterior1c,
                'controlRep2':gPosterior2c,
                'longRep1':gPosterior1l,
                'longRep2':gPosterior2l }

    tac = timeit.default_timer()
    print(">>> summary computation time:", (tac-tic)/60, "minutes") if (verbo==True) else None

    return gPosterior, cfMeanij, stars
