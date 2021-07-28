#/usr/bin/python3

"""
Functions to load and transform posterior distributions
"""

import os, sys
import numpy
import pandas

import warnings


def posterior_from_samples(samples, cmdstanDict):
    """ wrapper to extract posterior samples from cmdStan `sample` object, and reshape first axis with number of samples (S) and chains (L) final shape=(S,L) """

    cmdtraces = samples.stan_variables()
    hmc = samples.sampler_variables()

    chains = cmdstanDict['alice']

    dictraces = {}
    for key in cmdtraces.keys():

        dictraces[key] = reshape_chains(cmdtraces[key], chains)

    return dictraces, hmc


def reshape_chains(tracearray, chains):
    "Reshapes one-dimensional array with concatenated traces into bidimensional array with with number of samples (S) and chains (L) final shape=(S,L) "

    currentShape = list( tracearray.shape )
    newShape = [-1, chains] + currentShape[1:]

    newarray = tracearray.reshape( newShape )

    return newarray


def evorna_posteriors(folder, cmdstan_dict, fbidij, N, filenames={}, verbo=False):
    """
    Loads cmdStan trace files from disk, which must follow naming template: <modelname>_<selscheme><fbid1>_<fbid2>_nuts<iterations>_chain-<l>.csv
    """
    fbidi, fbidj = fbidij
    M = 2; trilM = 1

    if ( filenames == {} ):
        shortName = cmdstan_dict['modelname'] + "_short" + fbidi + "_" + fbidj + "_nuts" + str(cmdstan_dict['iterations']) + "_chain"
        controlName = cmdstan_dict['modelname'] + "_control" + fbidi + "_" + fbidj + "_nuts" + str(cmdstan_dict['iterations']) + "_chain"
        longName = cmdstan_dict['modelname'] + "_long" + fbidi + "_" + fbidj + "_nuts" + str(cmdstan_dict['iterations']) + "_chain"
    else:
        shortName = filenames["short"]
        controlName = filenames["control"]
        longName = filenames["long"]

    dictraceShort, hmcShort = read_gaussianprocess_traces(folder, shortName, cmdstan_dict, M, N, trilM, "short", muvar=muFlag, verbose=verbo)

    dictraceControl, hmcControl = read_gaussianprocess_traces(folder, controlName, cmdstan_dict, M, N, trilM, "control", muvar=muFlag, verbose=verbo)

    dictraceLong, hmcLong = read_gaussianprocess_traces(folder, longName, cmdstan_dict, M, N, trilM, "long", muvar=muFlag, verbose=verbo)

    return [[dictraceShort, hmcShort],[dictraceControl, hmcControl],[dictraceLong, hmcLong]]


def read_gaussianprocess_traces(folderpath, basefilename, cmd_dict, M, N, trilM, group="", remove=[], muvar=False, gaussian=True, verbo=False):

    modelName = cmd_dict['modelname']
    mcmc_algorithm = cmd_dict['mcmc_submethod']
    iterations = cmd_dict['iterations']
    alice = cmd_dict['alice']

    burnin = cmd_dict['burnin'] if cmd_dict['washburn'] else 0
    thinning = cmd_dict['thinning']

    thinterations = int(iterations/thinning)
    burnthin = int((burnin)/thinning)
    totalsamples = burnthin + thinterations

    namelist = [ os.path.join(folderpath, basefilename + str(l+1) + ".csv") for l in range(alice) if l not in remove ]

    listofTraceFrames = []
    for i,name in enumerate(namelist):
        if verbo:
            print( "\n>>> loading chain #" + str(i+1) + ": " + name)

        try:
            alicechain = pandas.read_csv( name, header=0, comment="#")
            if verbo:
                print( ">>> total number of samples in data frame: ", alicechain.shape[0])

            if (alicechain.shape[0] == totalsamples):
                 listofTraceFrames.append(alicechain)
            else:
                warnString = "<" + basefilename + str(i+1) + ">" + " does not have expected number of iterations and will be padded with missing values"
                warnings.warn(warnString, UserWarning)
                newFrame = pandas.DataFrame( index=range( 0, totalsamples ), columns=alicechain.columns )
                newFrame.iloc[0:alicechain.shape[0],:] = alicechain
                listofTraceFrames.append(newFrame)
        except:
            warnString = "<" + basefilename + str(i+1) + ">" + " doesn't exist, or loading/processing csv file raised errors -- skipping file"
            warnings.warn(warnString, UserWarning)

    posteriorDict, hmcDict = posterior_from_frame_list2(listofTraceFrames)

    return posteriorDict, hmcDict


def posterior_from_frame_list2(traceFramelist, M, N, trilM, verbose=False):
    """ gets sampler parameters traces (and puts the into a dictionary) and model parameter traces (and puts them into an array)"""

    iterations, framewitdh = traceFramelist[0].shape  # get shape of dataframe with samples
    chains = len(traceFramelist)  # get number of chains


    lpArray = numpy.full( [iterations, chains] )
    acceptArray = numpy.full( [iterations, chains] )
    stepArray = numpy.full( [iterations, chains] )
    treeArray = numpy.full( [iterations, chains] )
    leapArray = numpy.full( [iterations, chains] )
    divergentArray = numpy.full( [iterations, chains] )
    energyArray = numpy.full( [iterations, chains] )

    kfDiagArray = numpy.full( [iterations, chains, M] )
    ellArray = numpy.full( [iterations, chains, M] )

    kfTrilArray = numpy.full( [iterations, chains, trilM] )

    f1tilArray = numpy.full( [iterations, chains, M*N] )
    f2tilArray = numpy.full( [iterations, chains, M*N] )

    alphArray = numpy.full( [iterations, chains, M] )
    murray = numpy.full( [iterations, chains, M] )
    sigmArray = numpy.full( [iterations, chains, M] )

    for l, frameL in enumerate(traceFramelist):
        lpArray[:,l] = frameL['lp__'].values
        acceptArray[:,l] = frameL['accept_stat__'].values
        stepArray[:,l] = frameL['stepsize__'].values
        treeArray[:,l] = frameL['treedepth__'].values
        leapArray[:,l] = frameL['n_leapfrog__'].values
        divergentArray[:,l] = frameL['divergent__'].values
        energyArray[:,l] = frameL['energy__'].values

        kfDiagArray[:,l,:] = frameL.loc[:'KfDiag.1':'KfDiag.'+str(M)].values

        ellArray[:,l,:] = frameL.loc[:'ell.1':'ell.'+str(M)].values

        kfTrilArray[:,l,:] = frameL.loc[:'KfDiag.1'].values if (trilM==1) else frameL.loc[:'KfDiag.1':'KfDiag.'+str(trilM)].values

        f1tilArray[:,l,:] = frameL.loc[:'f1_til.1':'f1_til.'+str(M*N)].values
        f2tilArray[:,l,:] = frameL.loc[:'f2_til.1':'f2_til.'+str(M*N)].values

        if 'alpha.1' in frameL.columns:
            alphArray[:,l,:] = frameL.loc[:'alpha.1':'alpha.'+str(M)].values
        if 'sigma.1' in frameL.columns:
            sigmArray[:,l,:] = frameL.loc[:'sigman.1':'sigman.'+str(M)].values
        if 'mu.1' in frameL.columns:
            murray[:,l,:] = frameL.loc[:'mu.1':'mu.'+str(M)].values

    hmcDict = {
    'lp__':lpArray,
    'accept_stat__':acceptArray,
    'stepsize__':stepArray,
    'treedepth__':treeArray,
    'n_leapfrog__':leapArray,
    'divergent__':divergentArray,
    'energy__':energyArray
    }

    posteriorDict = {
    'KfDiag':kfDiagArray,
    'ell':ellArray,
    'KfTril':kfTrilArray,
    'f1_til':f1tilArray,
    'f2_til':f2tilArray,
    }

    if 'alpha.1' in traceFramelist[0].columns:
        posteriorDict['alpha'] = alphArray
    if 'sigma.1' in traceFramelist[0].columns:
        posteriorDict['sigma'] = sigmArray
    if 'mu.1' in traceFramelist[0].columns:
        posteriorDict['mu'] = murray

    return posteriorDict, hmcDict


def compute_Cf_ij(dictraces, fbidij=["",""], burn=0, verbo=False):
    """ Normalizes signal covariances in multichannel GP models with its signal variances (i.e. computes a correlation for a Gaussian Process) """

    fbidi, fbidj = fbidij
    print(">>> genes:", fbidi, "/", fbidj) if (verbo==True) else None

    dictraceShort, dictraceControl, dictraceLong = dictraces

    # compute posteriors for correlations in Multi-Channel Gaussian Process
    cfShortij = dictraceShort['KfTril'][burn:,:,0] / numpy.sqrt(dictraceShort['KfDiag'][burn:,:,0] * dictraceShort['KfDiag'][burn:,:,1] )
    cfControlij = dictraceControl['KfTril'][burn:,:,0] / numpy.sqrt( dictraceControl['KfDiag'][burn:,:,0] * dictraceControl['KfDiag'][burn:,:,1] )
    cfLongij = dictraceLong['KfTril'][burn:,:,0] / numpy.sqrt(dictraceLong['KfDiag'][burn:,:,0] * dictraceLong['KfDiag'][burn:,:,1] )

    # compute summaries from posterior
    cfShortMeanij = numpy.mean( cfShortij, axis=(0,1) )
    cfControlMeanij = numpy.mean( cfControlij, axis=(0,1) )
    cfLongMeanij = numpy.mean( cfLongij, axis=(0,1) )

    cfShort25ij = numpy.percentile( cfShortij, 2.5, axis=(0,1) )
    cfControl25ij = numpy.percentile( cfControlij, 2.5, axis=(0,1) )
    cfLong25ij = numpy.percentile( cfLongij, 2.5, axis=(0,1) )

    cfShort975ij = numpy.percentile( cfShortij, 97.5, axis=(0,1) )
    cfControl975ij = numpy.percentile( cfControlij, 97.5, axis=(0,1) )
    cfLong975ij = numpy.percentile( cfLongij, 97.5, axis=(0,1) )

    # create diciontaries with the summaries
    cfMeanij = {
            "short": cfShortMeanij,
            "control": cfControlMeanij,
            "long": cfLongMeanij
             }
    cf25ij = {
            "short": cfShort25ij,
            "control": cfControl25ij,
            "long": cfLong25ij
            }
    cf975ij = {
            "short": cfShort975ij,
            "control": cfControl975ij,
            "long": cfLong975ij
            }

    return cfMeanij, cf25ij, cf975ij


def overlap_cf_ij(cfMeanij, cf25ij, cf975ij):
    """ checks for overlap in the posteriors between experimental and control groups and assigns a start if there is no overlap """

    meanShort, meanControl, meanLong = cfMeanij['short'], cfMeanij['control'], cfMeanij['long']
    p25Short, p25Control, p25Long = cf25ij['short'], cf25ij['control'], cf25ij['long']
    p975Short, p975Control, p975Long = cf975ij['short'], cf975ij['control'], cf975ij['long']

    # short
    if ( (meanShort > meanControl) and (p25Short < p975Control) ):
        significanceShort = " (n.s.)"
    elif ( (meanShort < meanControl) and (p975Short > p25Control) ):
        significanceShort = " (n.s.)"
    else:
        significanceShort = "*"

    # long
    if ( (meanLong > meanControl) and (p25Long < p975Control) ):
        significanceLong = " (n.s.)"
    elif ( (meanLong < meanControl) and (p975Long > p25Control) ):
        significanceLong = " (n.s.)"
    else:
        significanceLong = "*"

    return significanceShort, significanceLong


def sel_summarize_gp_dictionary(fbid, seltraces):
    """ computes expectation and standard deviation from posterior for single channel parameters """

    singleChannelSummaries = ["kfDiagMean", "kfDiagSigma", "ellMean", "ellSigma", "alphaMean", "alphaSigma"]

    pairiors = pandas.DataFrame( index=[fbid], columns=singleChannelSummaries )

    axverage = tuple( range( len( seltraces['KfDiag'].shape ) ) )

    pairiors["kfDiagMean"] = numpy.mean( seltraces['KfDiag'], axis=axverage )
    pairiors["kfDiagSigma"] = numpy.std(seltraces['KfDiag'], axis=axverage )
    pairiors["ellMean"] = numpy.mean( seltraces['ell'], axis=axverage )
    pairiors["ellSigma"] = numpy.std(seltraces['ell'], axis=axverage )
    pairiors["alphaMean"] = numpy.mean( seltraces['alpha'], axis=axverage )
    pairiors["alphaSigma"] = numpy.std(seltraces['alpha'], axis=axverage )

    return pairiors


def summarize_gp_dictionary(fbid, dictraces):
    """ computes expectation and standard deviation from posterior for single channel parameters for all selection schemes"""

    dictraceShort, dictraceControl, dictraceLong = dictraces

    singleChannelSummaries = ["shortKfDiagMean", "shortKfDiagSigma", "shortEllMean", "shortEllSigma", "shortAlphaMean", "shortAlphaSigma", "controlKfDiagMean", "controlKfDiagSigma", "controlEllMean", "controlEllSigma", "controlAlphaMean", "controlAlphaSigma", "longKfDiagMean", "longKfDiagSigma", "longEllMean", "longEllSigma", "longAlphaMean", "longAlphaSigma"]

    pairiors = pandas.DataFrame( index=[fbid], columns=singleChannelSummaries )

    pairiors.loc[:, "shortKfDiagMean":"shortAlphaSigma"] = sel_summarize_gp_dictionary(fbid, dictraceShort).values
    pairiors.loc[:, "controlKfDiagMean":"controlAlphaSigma"] = sel_summarize_gp_dictionary(fbid, dictraceControl).values
    pairiors.loc[:, "longKfDiagMean":"longAlphaSigma"] = sel_summarize_gp_dictionary(fbid, dictraceLong).values

    return pairiors
