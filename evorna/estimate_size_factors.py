#!/usr/bin/python

import numpy, pandas
from scipy.stats.mstats import gmean as geomean

def deseq2(df):
    """
    'RLE'-like normalization method. Only uses variables (genes) where all observations (samples) has value nonzero value.
    If all genes have at least one zero raises ZeroDivisionError
    """
    # convert data frame to genes x samples format for computing purposes
    tRawCounts = pandas.DataFrame(df.T)
    tRawNonzeroCounts = tRawCounts[(tRawCounts.T > 0).all()]

    if (0 in tRawNonzeroCounts.shape):
        ZeroDivisionError(">> every gene contains at least one zero, geometric means will be all zero (or somehow there are no samples)")
    if (df.shape[0]>df.shape[1]):
        warnstring = ">>> Number of samples greater than number of genes; \n" + ">> make sure rows indicate samples and columns indicate genes."
        warnings.warn(warnstring, UserWarning)

    # compute geometric means aggregating over samples dimension
    sampleGeometricMeans = geomean(tRawNonzeroCounts, axis=1)

    # compute median (aggregating over gene dimension) of expression array after division by geometric means
    sizeFactors = numpy.median(tRawNonzeroCounts.div(sampleGeometricMeans, axis=0), axis=0)

    return sizeFactors


def deseq2_ignore_zeros(df):
    """
    'RLE'-like normalization method. Computes geometric means for genes with at least one nonzero sample, i.e. ignores zero values for that gene and computes geometric mean for the remaining values.
    """
    print(">> computing normalization factors by ignoring zero values.")

    tRawCounts = pandas.DataFrame(df.T)
    tRawAnyzeroCounts = tRawCounts[(tRawCounts.T > 0).any()]
    if (0 in tRawAnyzeroCounts.shape):
        ZeroDivisionError(">> all genes contains  are all zero (or somehow there are no samples)")

        return numpy.NaN

    elif (tRawCounts.shape[0]>tRawCounts.shape[1]):
        print(">> number of samples greater than number of genes;")
        print(">> make sure rows indicate samples and columns indicate genes.")

    yzer = numpy.zeros(tRawAnyzeroCounts.shape[0])
    for i in range(len(yzer)):

        geneslice = tRawAnyzeroCounts.iloc[i,:]
        yi = geneslice[geneslice>0]

        if yi.values.tolist()==[]:
            print(">> no non-zero values in gene", i)

        yzer[i] = geomean(yi)

    sizeFactors2 = numpy.median(tRawAnyzeroCounts.div(yzer, axis=0), axis=0)

    return sizeFactors2


def deseq2_add_one(df):
    """
    'RLE'-like normalization method with rough zero-value correction; removes zeros by adding one to all cells in data frame
    """

    df = pandas.DataFrame(df.T)

    if (df.shape[0]>df.shape[1]):
        print(">> WARNING: number of samples greater than number of genes;")
        print(">> make sure rows indicate samples and columns indicate genes.")

        warnstring = ">>> Number of samples greater than number of genes; \n" + ">> make sure rows indicate samples and columns indicate genes."
        warnings.warn(warnstring, UserWarning)

    tRawCounts = df.values.T

    for i in range(tRawCounts.shape[0]):
        for j in range(tRawCounts.shape[1]):
            if tRawCounts[i, j] == 0:
                tRawCounts[i, j] = 1

    tRawCountsPlusOne=tRawCounts+1

    sampleGeometricMeans = geomean(df.values, axis=1)  # compute geometric means over samples dimension

    sizeFactors3 = numpy.median((tRawCounts/sampleGeometricMeans).T, axis=0)

    return sizeFactors3


# DESeq2 R code
"""
estimateSizeFactorsForMatrix <- function(counts, locfunc=stats::median,
                                         geoMeans, controlGenes) {
  if (missing(geoMeans)) {
    incomingGeoMeans <- FALSE
    loggeomeans <- rowMeans(log(counts))
  } else {
    incomingGeoMeans <- TRUE
    if (length(geoMeans) != nrow(counts)) {
      stop("geoMeans should be as long as the number of rows of counts")
    }
    loggeomeans <- log(geoMeans)
  }
  if (all(is.infinite(loggeomeans))) {
    stop("every gene contains at least one zero, cannot compute log geometric means")
  }
  sf <- if (missing(controlGenes)) {
    apply(counts, 2, function(cnts) {
      exp(locfunc((log(cnts) - loggeomeans)[is.finite(loggeomeans) & cnts > 0]))
    })
  } else {
    if ( !( is.numeric(controlGenes) | is.logical(controlGenes) ) ) {
      stop("controlGenes should be either a numeric or logical vector")
    }
    loggeomeansSub <- loggeomeans[controlGenes]
    apply(counts[controlGenes,,drop=FALSE], 2, function(cnts) {
      exp(locfunc((log(cnts) - loggeomeansSub)[is.finite(loggeomeansSub) & cnts > 0]))
    })
  }
  if (incomingGeoMeans) {
    # stabilize size factors to have geometric mean of 1
    sf <- sf/exp(mean(log(sf)))
  }
  sf
}
"""
