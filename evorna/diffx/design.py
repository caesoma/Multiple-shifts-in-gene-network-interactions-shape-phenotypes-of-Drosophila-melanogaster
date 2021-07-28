#/usr/bin/python3
import os
import numpy
import pandas
import multiprocessing
import scipy.stats


def check_metadata_labels(data, metaData):
    """ check labels in data and metadata matrices match and if not return an empty object """
    if all(metaData.index != data.index):
        print(">> design and data labels do not match: check data")
        return None
    else:
        print(">> design and data labels are a match")
        return data, metaData


def check_design_labels(data, design, reDesign):
    """ check that design and reduced design matrices labels match data labels """
    if any(design.index != data.index):
        print(">> design and data labels do not match: check data")
        return None, None
    elif any(reDesign.index != data.index):
        print(">> design and data labels do not match: check data")
        return None, None
    else:
        print(">> design and data labels are a match")
        return design, reDesign


def create_design_matrix(metadata):
    """ create sex-specific "design matrix", i.e. specify coefficients of sexes-separate linear model from experimental design variables (specific to the variables in Souto-Maior et al. [2021] ) """
    print(">>> creating design matrix...")

    design = pandas.DataFrame(0, index=metadata.index, columns=["bIntercept1", "bIntercept2", "bShort1", "bShort2", "bLong1", "bLong2", "generation", "bShortGen1", "bShortGen2", "bLongGen1", "bLongGen2"])

    # one of two different intercept terms are given for all observations, conditional on their experimental replicate
    design.loc[ ( metadata['Rep'] == 1 ), 'bIntercept1'] = 1
    design.loc[ ( metadata['Rep'] == 2 ), 'bIntercept2'] = 1

    # treatment-dependent, replicate-specific intercept terms are given to each sample (not including controls)
    for label in metadata.index:
        if ( (metadata.loc[label, 'sel'] == "short") & (metadata.loc[label, 'Rep'] == 1) ):
            design.loc[label, 'bShort1'] = 1
        elif ( (metadata.loc[label, 'sel'] == "short") & (metadata.loc[label, 'Rep'] == 2) ):
            design.loc[label, 'bShort2'] = 1
        elif ( (metadata.loc[label, 'sel'] == "long") & (metadata.loc[label, 'Rep'] == 1) ):
            design.loc[label, 'bLong1'] = 1
        elif ( (metadata.loc[label, 'sel'] == "long") & (metadata.loc[label, 'Rep']==2) ):
            design.loc[label, 'bLong2'] = 1

    # generation-dependent slope is given to all posteriorFull -- treatment-dependent, replicate-specific interaction slopes are are assigned each treatment (as above, excluding controls)
    design['generation'] = metadata['generation']
    design['bShortGen1'] = metadata['generation'] * design['bShort1']
    design['bShortGen2'] = metadata['generation'] * design['bShort2']
    design['bLongGen1']  = metadata['generation'] * design['bLong1']
    design['bLongGen2']  = metadata['generation'] * design['bLong2']

    # reduced design excludes
    reduceDesign = design[ ["bIntercept1", "bIntercept2", "bShort1", "bShort2", "bLong1", "bLong2", "generation"] ]

    return design, reduceDesign


def create_results_frames(data, flyBaseInfo=[], verbo=False):

    print(">>> Creating data frame:") if (verbo==True) else None

    N = data.shape[0]

    paramcols = ["bIntercept1", "bIntercept2", "bShort1", "bShort2", "bLong1", "bLong2", "gen", "bShortGen1", "bShortGen2", "bLongGen1", "bLongGen2", "dispersion", "flogp", "rIntercept1", "rIntercept2", "rShort1", "rShort2", "rLong1", "rLong2", "redgen", "redispersion", "rlogp", "p", "padj", "muShort", "sigmaShort", "muControl", "sigmaControl", "muLong", "sigmaLong", "muShortGen", "sigmaShortGen", "muLongGen", "sigmaLongGen", "ruShort", "rigmaShort", "ruControl", "rigmaControl", "ruLong", "rigmaLong"]

    outputcols = ['fullMean.'+str(i) for i in range(N)] + ['full25.'+str(i) for i in range(N)] + ['full975.'+str(i) for i in range(N)] + ['reducedMean.'+str(i) for i in range(N)] + ['reduced25.'+str(i) for i in range(N)] + ['reduced975.'+str(i) for i in range(N)]

    if ( len(data.shape)==2 ):
        resultsFrame = pandas.DataFrame( index=data.columns, columns=paramcols+outputcols, dtype=numpy.int64)
        symbolFrame = pandas.DataFrame( index=data.columns, columns=["symbol"])
    elif ( len(data.shape)==1 ):
        resultsFrame = pandas.DataFrame( index=[data.name], columns=paramcols+outputcols, dtype=numpy.int64)
        symbolFrame = pandas.DataFrame( index=[data.name], columns=["symbol"])

    print(">>> adding gene symbols matching FBIDs:") if (verbo==True) else None

    if ( len(flyBaseInfo)>0 ):
        for fbid in resultsFrame.index:
            print( fbid, ":", flyBaseInfo.loc[fbid, 'SYMBOL'], sep='', end=', ' ) if (verbo==True) else None
            symbolFrame.loc[fbid, 'symbol'] = flyBaseInfo.loc[fbid, 'SYMBOL']

    return symbolFrame, resultsFrame


def split_results_frame(metadata, resultsFrame, results_dir=""):

    """ Splits and organizes results data frame into parameters/summaries and model predictions (mean and confidence limits) """

    parameterFrame = resultsFrame.loc[:, :"rigmaLong"]

    fullPosteriorMeanFrame = pandas.DataFrame(index=metadata.index, columns=resultsFrame.index)
    fullPosterior25Frame = pandas.DataFrame(index=metadata.index, columns=resultsFrame.index)
    fullPosterior975Frame = pandas.DataFrame(index=metadata.index, columns=resultsFrame.index)

    reducedPosteriorMeanFrame = pandas.DataFrame(index=metadata.index, columns=resultsFrame.index)
    reducedPosterior25Frame = pandas.DataFrame(index=metadata.index, columns=resultsFrame.index)
    reducedPosterior975Frame = pandas.DataFrame(index=metadata.index, columns=resultsFrame.index)

    N = metadata.shape[0]

    fbid0 = resultsFrame.index[0]
    fbidN = resultsFrame.index[-1]

    fullPosteriorMeanFrame.loc[:,:] = resultsFrame.loc[ fbid0:fbidN, "fullMean.0":"fullMean."+str(N-1) ].T.values
    fullPosterior25Frame.loc[:,:] = resultsFrame.loc[ fbid0:fbidN, "full25.0":"full25."+str(N-1) ].T.values
    fullPosterior975Frame.loc[:,:] = resultsFrame.loc[ fbid0:fbidN, "full975.0":"full975."+str(N-1) ].T.values

    reducedPosteriorMeanFrame.loc[:,:] = resultsFrame.loc[ fbid0:fbidN, "reducedMean.0":"reducedMean."+str(N-1) ].T.values
    reducedPosterior25Frame.loc[:,:] = resultsFrame.loc[ fbid0:fbidN, "reduced25.0":"reduced25."+str(N-1) ].T.values
    reducedPosterior975Frame.loc[:,:] = resultsFrame.loc[ fbid0:fbidN, "reduced975.0":"reduced975."+str(N-1) ].T.values

    if (results_dir != ""):
        parameterFrame.to_csv( "parameter data frame.csv", sep=",", header=True, index=True )

        fullPosteriorMeanFrame.to_csv( os.path.join( results_dir, "full posterior mean.csv") , sep=",", header=True, index=True )
        fullPosterior25Frame.to_csv( os.path.join( results_dir, "full posterior 25p.csv"), sep=",", header=True, index=True )
        fullPosterior975Frame.to_csv( os.path.join( results_dir, "full posterior 25p.csv"), sep=",", header=True, index=True )

        reducedPosteriorMeanFrame.to_csv( os.path.join( results_dir, "reduced posterior mean.csv"), sep=",", header=True, index=True )
        reducedPosterior25Frame.to_csv( os.path.join( results_dir, "reduced posterior 975p.csv"), sep=",", header=True, index=True )
        reducedPosterior975Frame.to_csv( os.path.join( results_dir, "reduced posterior 25p.csv"), sep=",", header=True, index=True )

    return parameterFrame, fullPosteriorMeanFrame, fullPosterior25Frame, fullPosterior975Frame, reducedPosteriorMeanFrame, reducedPosterior25Frame, reducedPosterior975Frame
