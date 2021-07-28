#/usr/bin/python3
import os
import numpy
import pandas
import multiprocessing
import scipy.stats
import warnings

import timeit

import stan


from evorna.diffx import design, model


def stan_dictionary(fbid, data, metadata, designFrame, reducedDesign, tinyDesign=[]):
    """ assembles Stan "data" dictionary """

    stanDictionary = {}
    stanDictionary['N'] = N = designFrame.shape[0]

    stanDictionary['P'] = P = designFrame.shape[1]
    stanDictionary['R'] = R = reducedDesign.shape[1]

    if ( len(tinyDesign)>0 ):
        stanDictionary['T'] = tinyDesign.shape[1]
        stanDictionary['tinyMatrix'] = tinyDesign.values

    stanDictionary['designMatrix'] = designFrame.values
    stanDictionary['reducedMatrix'] = reducedDesign.values
    #

    stanDictionary['y'] = data[fbid].values
    stanDictionary['sfs'] = sizeFactors = metadata.sizeFactors.values

    stanDictionary['printercept'] = max( 1e-9, numpy.log( numpy.mean( data.loc[metadata.generation == 0, fbid].values ) ) ) # log of observations means at generation zero
    stanDictionary['stdcept'] = max( 1, numpy.log( numpy.std( data.loc[metadata.generation == 0, fbid].values ) ) ) # log of observations standard deviation at generation zero (cannot be negative, and is limited to 1 at least) -- is not used in model used for publication, where it's fixed at 1

    return stanDictionary


def pyStanModel(model_path, modelName="stan_model", dataDictionary={}, verbo=False):
    """ wrapper function to compile PyStan model from model path (and model name) """

    with open(model_path, 'r') as stanfile:
        model_file = stanfile.read()

    print(">>> Python passing it on to C++ to do its thing...\n") if (verbo==True) else None

    if (dataDictionary == {}):
        model = pystan.StanModel( model_code=model_file, model_name=modelName )
    else:
        model = stan.build( program_code=model_file, data=dataDictionary )

    return model


def assemble_parallel_results(data, output, flyBaseInfo):
    """ a function to assemble the results of a parallel run output into a results data frame """
    symbolFrame, resultsFrame = design.create_results_frames(data, flyBaseInfo, verbo=verbo )

    fbidList = list( resultsFrame.index )

    print(">>> building data frame from multiprocessing output...")
    for i, fbid in enumerate(fbidList):
        resultsFrame.loc[ fbid, : ] = output[i]

    return symbolFrame, resultsFrame


def fit_parallel(fbidList, data, metadata, designFrame, reducedDesign, flyBaseInfo, models, commands={}, cores=8, verbo=True):
    """ a function to use the `multiprocessing.map` function to fit genes in parallel
    WARNING: this function is untested with PyStan 3 and may not work, it is provided here as a sort of pseudocode to parallelize the runs, but the guaranteed approach to parallelize several gene runs is to set up external cmdStan runs (using the model externally compiled, like the one used by the `hierarchical_glm` module)
    """

    full_model_path, reduced_model_path = models

    stanDictionary = stan_dictionary(fbidList[0], data, metadata, designFrame, reducedDesign)

    full_model = pyStanModel(model_path=full_model_path, dataDictionary=stanDictionary, verbo=False)
    reduced_model = pyStanModel(model_path=reduced_model_path, dataDictionary=stanDictionary, verbo=False)

    tupplist = zip( fbidList, len(fbidList)*[data], len(fbidList)*[metadata], len(fbidList)*[designFrame], len(fbidList)*[reducedDesign], len(fbidList)*[[full_model_path, reduced_model_path]], len(fbidList)*[os.path.join( results_dir, "traces")], len(fbidList)*["py"] )

    tic = timeit.default_timer()
    print(">>> Fitting GLMs using multiprocessing Pool module:") if (verbo==True) else None
    with multiprocessing.Pool(processes=cores) as parallel:
        output = parallel.map( fit_tupple, tupplist )
    tac = timeit.default_timer()
    print(">>> parallel fit run time:", tac-tic, "seconds \n") if (verbo==True) else None

    symbolFrame, resultsFrame = assemble_parallel_results(data, output, flyBaseInfo)

    return symbolFrame, resultsFrame


def fit_tupple(arguments):
    """ helper function to allow tupple or arguments to be used with `fit` function, below (needed for `Pool` map function over a generator/list of tuples)"""

    fbid, data, metadata, designFrame, reducedDesign, models = arguments

    return fit( fbid, data, metadata, designFrame, reducedDesign, models )


def fit(fbid, data, metadata, designFrame, reducedDesign, models, commands={}, verbo=False):  # , tinyDesign):
    """ Hierarchical generalized linear model (HGLM) fit using CmdStanPy interface (writes posteriors/traces to working directory) """

    y = data[fbid].values

    # The following command shows what the index number of the gene is if the entire count data frame needs is passed to the function, otherwise it will be "1 of 1"
    print("(", data.columns.tolist().index(fbid), "of", data.shape[1], ")") if (verbo==True) else None

    print(">>> creating Stan data dictionaries") if (verbo==True) else None
    # Stan model
    stanDictionary = stan_dictionary(fbid, data, metadata, designFrame, reducedDesign, tinyDesign=[])

    N = stanDictionary['N']
    P = stanDictionary['P']
    R = stanDictionary['R']

    sizeFactors = stanDictionary['sfs']

    # some trivial but likely nonzero initial values to avoid low-probability regions
    init_dict = {}
    init_dict['bIntercept1'] = stanDictionary['printercept']
    init_dict['bIntercept2'] = stanDictionary['printercept']

    rinit_dict = {}
    rinit_dict['bIntercept1'] = stanDictionary['printercept']
    rinit_dict['bIntercept2'] = stanDictionary['printercept']

    #tinit_dict = {}
    #tinit_dict['bIntercept1'] = stanDictionary['printercept']
    #tinit_dict['bIntercept2'] = stanDictionary['printercept']

    if (commands!={}):
        alice = 1  # number of chains
        iterations = commands["iterations"] # number of iterations
        burnin = commands["burnin"] # number of burn-in samplesFull
        thinlength = commands["thinlength"] # number of samplesFull kept after running whole chain
        thinning = int( max(1, ( iterations - burnin) / thinlength ) )
    else:
        alice = 1  # number of chains
        iterations = 2000
        burnin = 2000
        thinlength = 1000
        thinning = int( max(1, ( iterations - burnin) / thinlength ) )

    full_model_path, reduced_model_path = models # models must be pystan model objects

    full_model = pyStanModel( full_model_path, modelName="full_model", dataDictionary=stanDictionary, verbo=False)
    reduced_model = pyStanModel( reduced_model_path, modelName="reduced_model", dataDictionary=stanDictionary, verbo=False)

    _, resultsj = design.create_results_frames( data[fbid], flyBaseInfo=[], verbo=False )

    print("gene:", resultsj.loc[fbid].symbol, end=', ') if (verbo==True) else None
    print("mean expression:", numpy.round( numpy.mean(y), 0), end=', ') if (verbo==True) else None

    # `try` statement will make sure a loop using this function will not be broken, but may return empty results for some genes in an otherwise succesful run (see `except` case below)
    # try:
    # MCMC inference using Stan model
    samplesFull = full_model.sample(num_samples=iterations, num_warmup=burnin, save_warmup=False, init=alice*[init_dict], num_thin=thinning, num_chains=alice)

    samplesReduced = reduced_model.sample(num_samples=iterations, num_warmup=burnin, save_warmup=False, init=alice*[init_dict], num_thin=thinning, num_chains=alice)
    # tinypling = tinyStan.sample(data=stanDictionary, num_samples=iterations, num_warmup=burnin, save_warmup=False, init=alice*[init_dict], num_thin=thinning, num_chains=alice)

    # extract posterior MCMC samplesFull from PyStan object
    posteriorFull = samplesFull.to_frame()
    posteriorReduced = samplesReduced.to_frame()
    # posteriorTiny = tinypling.stan_variables()

    # index with maximum a posteriori probability (MAP)
    mapindex = list( posteriorFull['lp__'] ).index( posteriorFull['lp__'].max() )
    redmapindex = list( posteriorReduced['lp__'] ).index( posteriorReduced['lp__'].max() )
    # tinindex = posteriorTiny['lp__'].tolist().index(posteriorTiny['lp__'].max())

    # likelihood recomputed at MAP
    flogp = model.negative_binomial_loglikelihood( mean=samplesFull['full'][:,mapindex]*sizeFactors, dispersion=posteriorFull['alpha'][mapindex], observed=y )
    rlogp = model.negative_binomial_loglikelihood( mean=samplesReduced['reduced'][:,redmapindex]*sizeFactors, dispersion=posteriorReduced['alpha'][redmapindex], observed=y )

    #tlogp = model.negative_binomial_loglikelihood( mean=posteriorTiny['tiny'][tinindex]*sizeFactors, dispersion=posteriorTiny['alpha'][tinindex], observed=y )

    # likelihood ratio test
    pvalue, lratio = model.likelihood_ratio_test( flogp, rlogp, (designFrame.shape[1] - reducedDesign.shape[1]) )

    # assigning looks ugly like this, but it's too long to make it one line per assignment
    resultsj.loc[fbid, 'bIntercept1'], resultsj.loc[fbid, 'bIntercept2'], resultsj.loc[fbid, 'bShort1'], resultsj.loc[fbid, 'bShort2'], resultsj.loc[fbid, 'bLong1'], resultsj.loc[fbid, 'bLong2'], resultsj.loc[fbid, 'gen'], resultsj.loc[fbid, 'bShortGen1'], resultsj.loc[fbid, 'bShortGen2'], resultsj.loc[fbid, 'bLongGen1'], resultsj.loc[fbid, 'bLongGen2'], resultsj.loc[fbid, 'dispersion'] = posteriorFull['bIntercept1'][mapindex], posteriorFull['bIntercept2'][mapindex], posteriorFull['bShort1'][mapindex], posteriorFull['bShort2'][mapindex], posteriorFull['bLong1'][mapindex], posteriorFull['bLong2'][mapindex], posteriorFull['gen'][mapindex], posteriorFull['bShortGen1'][mapindex], posteriorFull['bShortGen2'][mapindex], posteriorFull['bLongGen1'][mapindex], posteriorFull['bLongGen2'][mapindex], posteriorFull['alpha'][mapindex]

    resultsj.loc[fbid, 'rIntercept1'], resultsj.loc[fbid, 'rIntercept2'], resultsj.loc[fbid, 'rShort1'], resultsj.loc[fbid, 'rShort2'], resultsj.loc[fbid, 'rLong1'], resultsj.loc[fbid, 'rLong2'], resultsj.loc[fbid, 'redgen'], resultsj.loc[fbid, 'redispersion'] = posteriorReduced['bIntercept1'][redmapindex], posteriorReduced['bIntercept2'][redmapindex], posteriorReduced['bShort1'][mapindex], posteriorReduced['bShort2'][redmapindex], posteriorReduced['bLong1'][redmapindex], posteriorReduced['bLong2'][redmapindex], posteriorReduced['gen'][redmapindex], posteriorReduced['alpha'][redmapindex]

    # resultsj.loc[fbid, 'tIntercept1'], resultsj.loc[fbid, 'tIntercept2'], resultsj.loc[fbid, 'tingen'], resultsj.loc[fbid, 'tinispersion'] = posteriorTiny['bIntercept1'][tinindex], posteriorTiny['bIntercept2'][tinindex], posteriorTiny['gen'][tinindex], posteriorTiny['alpha'][tinindex]

    resultsj.loc[fbid, 'muShort'], resultsj.loc[fbid, 'sigmaShort'], resultsj.loc[fbid, 'muControl'], resultsj.loc[fbid, 'sigmaControl'], resultsj.loc[fbid, 'muLong'], resultsj.loc[fbid, 'sigmaLong'], resultsj.loc[fbid, 'muShortGen'], resultsj.loc[fbid, 'sigmaShortGen'], resultsj.loc[fbid, 'muLongGen'], resultsj.loc[fbid, 'sigmaLongGen'] = posteriorFull['muShort'][mapindex], posteriorFull['sigmaShort'][mapindex], posteriorFull['muControl'][mapindex], posteriorFull['sigmaControl'][mapindex], posteriorFull['muLong'][mapindex], posteriorFull['sigmaLong'][mapindex], posteriorFull['muShortGen'][mapindex], posteriorFull['sigmaShortGen'][mapindex], posteriorFull['muLongGen'][mapindex], posteriorFull['sigmaLongGen'][mapindex]

    resultsj.loc[fbid, 'ruShort'], resultsj.loc[fbid, 'rigmaShort'], resultsj.loc[fbid, 'ruControl'], resultsj.loc[fbid, 'rigmaControl'], resultsj.loc[fbid, 'ruLong'], resultsj.loc[fbid, 'rigmaLong'] = posteriorReduced['muShort'][redmapindex], posteriorReduced['sigmaShort'][redmapindex], posteriorReduced['muControl'][redmapindex], posteriorReduced['sigmaControl'][redmapindex], posteriorReduced['muLong'][redmapindex], posteriorReduced['sigmaLong'][redmapindex]

    # resultsj.loc[fbid, 'tuControl'], resultsj.loc[fbid, 'tigmaControl'] = posteriorTiny['muControl'][tinindex], posteriorTiny['sigmaControl'][tinindex]

    resultsj.loc[fbid, 'flogp'] = flogp
    resultsj.loc[fbid, 'rlogp'] = rlogp
    resultsj.loc[fbid, 'p'] = pvalue
    #resultsj.loc[fbid, 'tlogp'] = tlogp

    resultsj.loc[ :, "fullMean.0":"fullMean."+str(N-1) ] = samplesFull['full'].mean(axis=1)
    resultsj.loc[ :, "full25.0":"full25."+str(N-1) ] = numpy.percentile(samplesFull['full'], 2.5, axis=1)
    resultsj.loc[ :, "full975.0":"full975."+str(N-1) ]  = numpy.percentile(samplesFull['full'], 97.5, axis=1)

    resultsj.loc[ :, "reducedMean.0":"reducedMean."+str(N-1) ] = samplesReduced['reduced'].mean(axis=1)
    resultsj.loc[ :, "reduced25.0":"reduced25."+str(N-1) ] = numpy.percentile(samplesReduced['reduced'], 2.5, axis=1)
    resultsj.loc[ :, "reduced975.0":"reduced975."+str(N-1) ] = numpy.percentile(samplesReduced['reduced'], 97.5, axis=1)

    #resultsj.loc[ :, "tinyMean.0":"tinyMean."+str(N-1) ] = samplesReduced['tiny'].mean(axis=0)
    #resultsj.loc[ :, "tiny25.0":"tiny25."+str(N-1) ] = numpy.percentile(samplesReduced['tiny'], 2.5, axis=0)
    #resultsj.loc[ :, "tiny975.0":"tiny975."+str(N-1) ] = numpy.percentile(samplesReduced['tiny'], 97.5, axis=0)

    #except RuntimeError:
    #    print(">>> RuntimeError, will return empty array")

    return resultsj.loc[fbid]
