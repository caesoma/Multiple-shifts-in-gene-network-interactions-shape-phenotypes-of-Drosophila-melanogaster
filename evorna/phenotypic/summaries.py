#!/usr/bin/python3

"""
Provides functions for phenotypic analysis of artifical selection experiment
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "3.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import os
import numpy
import pandas

def raw(data_dir, dataFile):
    """ simple wrapper for reading csv file with header """
    data = pandas.read_csv( os.path.join( data_dir, dataFile), sep=",", header=0)  #, index_col=0 )
    return data


def meta_data(data_dir, dataFile, samples=True):
    """ reads data and metadata from DAM data """
    alldata = raw(data_dir, dataFile)

    dataColumns = ["boutd", "sleepd", "boutn", "sleepn", "wakeact", "sleeplat", "avgboutd", "avgboutn"]

    if (samples==False):
        metaColumns = ["Monitor", "Channel", "Sex", "Line", "sel", "Rep", "Generation", "Block", "Setup Code", "N Rows", "Disposition", "Actual_Disposition"]
    else:
        metaColumns = ["Sample_ID", "Monitor", "Channel", "Sex", "Line", "sel", "Rep", "Generation", "Block", "Setup Code", "N Rows", "Disposition", "Actual_Disposition", "sample_replicate"]

    metadata = alldata[metaColumns]
    data = alldata[dataColumns]

    return metadata, data


def design_subset(traitData, metadata, condition):
    """ get subset from trait data based on some condition (i.e. selection scheme, replicate and generation combination set) """

    sell, repp, genn = condition
    subset = traitData.loc[((metadata.sel == sell) & (metadata.Rep == repp) & (metadata.Generation == genn))]

    return subset


def trait_mean(traita, metadata, condlist):
    """ compute mean of traits under different conditions listed """
    f = lambda condition: numpy.mean( design_subset(traita, metadata, condition) )

    traitMean = list( map ( f, condlist ) )

    return traitMean


def trait_variance(traita, metadata, condlist):
    """ compute variance of traits under different conditions listed """
    f = lambda condition: numpy.var(design_subset(traita, metadata, condition))
    traitVariance = list( map ( f, condlist ) )

    return traitVariance


def trait_coefficient_of_variation(traita, metadata, condlist):
    """ compute coefficient of variation of traits under different conditions listed """

    traitMean = numpy.array( trait_mean(traita, metadata, condlist) )
    traitVariance = numpy.sqrt( trait_variance(traita, metadata, condlist) )

    traitCV = list( traitVariance/traitMean )

    return traitCV


def trait_mean_frame(data, metadata, traitlist):
    """ compute data frame with means of traits """

    # lists of design variables
    selist = sorted(set(metadata.sel))
    replist = sorted(set(metadata.Rep))
    genlist = sorted(set(metadata.Generation))

    # lengths from number of levels in each variable
    nSel = len(selist)
    nRep = len(replist)
    nGen = len(genlist)
    nTrait = len(traitlist)

    # new data frame with combinations of design variables
    meanDesign = pandas.DataFrame(index=range(nSel*nRep*nGen), columns=["Sel", "Rep", "Generation"])
    meanDesign.Sel = [sell for sell in selist for _ in range(nRep)]*nGen  #  nRep*nGen*selist
    meanDesign.Rep = nSel*nGen*replist
    meanDesign.Generation = [genn for genn in genlist for _ in range(nSel*nRep)]

    # list of conditions
    condlist = list( zip(meanDesign.Sel, meanDesign.Rep, meanDesign.Generation ) )

    # compute arrays with means and put them into new data frame with means
    f = lambda trait: trait_mean( data[trait], metadata, condlist )
    traitArray = numpy.transpose( list( map(f, traitlist) ) )
    traitMeans = pandas.DataFrame( traitArray, index=range(nSel*nRep*nGen), columns=traitlist )

    return pandas.concat( [meanDesign, traitMeans], axis=1 )


def trait_variance_frame(data, metadata, traitlist):
    """ same as `trait_mean_frame` but for variance """
    selist = sorted(set(metadata.sel))
    replist = sorted(set(metadata.Rep))
    genlist = sorted(set(metadata.Generation))
    nSel = len(selist)
    nRep = len(replist)
    nGen = len(genlist)
    nTrait = len(traitlist)

    varianceDesign = pandas.DataFrame(index=range(nSel*nRep*nGen), columns=["Sel", "Rep", "Generation"])
    varianceDesign.Sel = [sell for sell in selist for _ in range(nRep)]*nGen  #  nRep*nGen*selist
    varianceDesign.Rep = nSel*nGen*replist
    varianceDesign.Generation = [genn for genn in genlist for _ in range(nSel*nRep)]

    condlist = list( zip(varianceDesign.Sel, varianceDesign.Rep, varianceDesign.Generation ) )

    f = lambda trait: trait_variance( data[trait], metadata, condlist )
    traitArray = numpy.transpose( list( map(f, traitlist) ) )
    traitVariance = pandas.DataFrame( traitArray, index=range(nSel*nRep*nGen), columns=traitlist )

    return pandas.concat( [varianceDesign, traitVariance], axis=1 )


def trait_cv_frame(data, metadata, traitlist):
    """ same as `trait_mean_frame` but for coefficient of variation """
    selist = sorted(set(metadata.sel))
    replist = sorted(set(metadata.Rep))
    genlist = sorted(set(metadata.Generation))
    nSel = len(selist)
    nRep = len(replist)
    nGen = len(genlist)
    nTrait = len(traitlist)

    meanDesign = pandas.DataFrame(index=range(nSel*nRep*nGen), columns=["Sel", "Rep", "Generation"])
    meanDesign.Sel = [sell for sell in selist for _ in range(nRep)]*nGen  #  nRep*nGen*selist
    meanDesign.Rep = nSel*nGen*replist
    meanDesign.Generation = [genn for genn in genlist for _ in range(nSel*nRep)]

    condlist = list( zip(meanDesign.Sel, meanDesign.Rep, meanDesign.Generation ) )

    f = lambda trait: trait_coefficient_of_variation( data[trait], metadata, condlist )
    traitArray = numpy.transpose( list( map(f, traitlist) ) )
    traitCV = pandas.DataFrame( traitArray, index=range(nSel*nRep*nGen), columns=traitlist )

    return pandas.concat( [meanDesign, traitCV], axis=1 )


def loopy_trait_means(traitlist, metadata, data, generations):
    """ same as `trait_mean_frame` but using loops """
    selist = sorted(set(metadata.sel))
    replist = list(set(metadata.Rep))
    genlist = list(set(metadata.Generation))
    nSel = len(selist)
    nRep = len(replist)
    nGen = len(genlist)
    nTrait = len(traitlist)

    traitMeans = pandas.DataFrame(index=range(nSel*nRep*nGen), columns=["Sel", "Rep", "Generation"]+traitlist)
    traitMeans.Sel = [sell for sell in selist for _ in range(nRep)]*nGen  #  nRep*nGen*selist
    traitMeans.Rep = nSel*nGen*replist
    traitMeans.Generation = [genn for genn in genlist for _ in range(nSel*nRep)]

    for tt in traitlist:
        for sell in selist:
            for repp in replist:
                for genn in genlist:
                    traitMeans.loc[((traitMeans.Sel == sell) & (traitMeans.Rep == repp) & (traitMeans.Generation == genn)), tt] = numpy.mean(phenoData.loc[((metadata.sel == sell) & (metadata.Rep == repp) & (metadata.Generation == genn)), tt])

    return traitMeans


def breeders_delta(metadata, data, trait="sleepn", line="Line", generation="Generation", intermediates=False):
    """ compute creeders equation delta, the per-generation differentials """

    if line not in metadata.columns:
        metadata[line] = "line"

    nextgenmeans = line_nextgen_means(metadata, data, trait=trait, line=line, generation=generation)
    linemeans = line_means(metadata, data, trait=trait, line=line, generation=generation)

    #print(nextgenmeans)
    #print(linemeans)
    #print(nextgenmeans - linemeans)

    ΔS = nextgenmeans - linemeans
    ΔR = pandas.DataFrame(data = linemeans.loc[linemeans.index[1:]].values - linemeans.loc[linemeans.index[:-1]].values, index=[str(t+1) + "-" + str(t) for t in ΔS.index[:-1]] , columns = ΔS.columns)

    if (intermediates==False):
        return ΔS.loc[linemeans.index[0:-1],:], ΔR  # ΔS has N generations while ΔR has N-1, so the last differential is ommitted
    else:
        return ΔS.loc[linemeans.index[0:-1],:], ΔR, nextgenmeans, linemeans


def breeders_sigma(metadata, data, trait="sleepn", line="Line", generation="Generation", intermediates=False):
    """ compute creeders equation sigma, the cumulated differentials """
    if line not in metadata.columns:
        metadata[line] = "line"

    ΔS, ΔR, nextgenmeans, linemeans = breeders_delta(metadata, data, trait=trait, line=line, generation=generation, intermediates=True)

    ΣS = pandas.DataFrame(index=ΔS.index, columns=ΔS.columns)
    ΣR = pandas.DataFrame(index=ΔR.index, columns=ΔR.columns)

    for linn in ΔS.columns:
        ΣS.loc[:,linn] = numpy.cumsum( ΔS.loc[:,linn] )
        ΣR.loc[:,linn] = numpy.cumsum( ΔR.loc[:,linn] )

    if (intermediates==False):
        return ΣS, ΣR
    else:
        return ΣS, ΣR, ΔS, ΔR, nextgenmeans, linemeans



def line_means(metadata, data, trait="sleepn", line="Line", generation="Generation"):
    """ line mean traits at some generation """

    if line not in metadata.columns:
        metadata[line] = "line"

    linemeans = pandas.DataFrame(index=sorted( set(metadata[generation] ) ), columns=sorted( set( metadata[line]) ), dtype=float )

    for linn in set( metadata[line]):
        for genn in sorted( set( metadata[generation]) ):

            linemeans.loc[ genn, linn ] = numpy.mean(  data.loc[ ( ( metadata[line]==linn) & ( metadata[generation]==genn ) ), trait] )

    return linemeans


def line_nextgen_means(metadata, data, trait="sleepn", line="Line", generation="Generation"):
    """ line mean traits at next generation """

    if line not in metadata.columns:
        metadata[line] = "line"

    linemeans = pandas.DataFrame(index=sorted( set(metadata[generation] ) ), columns=sorted( set( metadata[line]) ), dtype=float )

    for linn in set( metadata[line]):
        for genn in sorted( set( metadata[generation]) ):

            linemeans.loc[ genn, linn ] = numpy.mean(  data.loc[ ( ( metadata[line]==linn) & ( metadata[generation]==genn ) & ( metadata.Actual_Disposition=="next_gen") ), trait] )

    return linemeans
