#!/usr/bin/python3

"""
Functions to compute correlations between channels form Gaussian Process (mostly untested within the package, so relative directory paths and filesystem-dependent functions may need review)
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import os, sys, platform
import numpy, numba
import pandas
import networkx
import json

import matplotlib, seaborn
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import figure, plot, show

import warnings

from evorna.nonlinear.plot_K_functions import plotK, plotK_all2, plotK_differences2


def write_json_file(filename, diciontary, idnt=None, printFlag=False):
    """ wrapper for writing json data """

    dump = json.dumps( diciontary, indent=idnt )
    if printFlag:
        print(dump)
    try:
        with open(filename, 'w') as fhandle:
            fhandle.write( dump )
    except:
        warnings.warn(">>> error writing file, check json dump", UserWarning)

    return dump

def load_cf(summaries_dir, M):
    """ funciton  for reading correlations (cf) limits, as well as R-hat and ESS metrics for covariance parameters (kf) and posteriors (lp) for M genes (file names must match names in function below) """

        cfMean = {}
        cf25 = {}
        cf975 = {}

        kfHatR = {}
        lpHatR = {}

        kfESS = {}
        lpESS = {}

        cfMean['short'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"ShortMean.csv") , sep=",", header=0, index_col=0)
        cfMean['control'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"ControlMean.csv") , sep=",", header=0, index_col=0)
        cfMean['long'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"LongMean.csv") , sep=",", header=0, index_col=0)

        cf25['short'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"Short25.csv") , sep=",", header=0, index_col=0)
        cf25['control'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"Control25.csv") , sep=",", header=0, index_col=0)
        cf25['long'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"Long25.csv") , sep=",", header=0, index_col=0)

        cf975['short'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"Short975.csv") , sep=",", header=0, index_col=0)
        cf975['control'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"Control975.csv") , sep=",", header=0, index_col=0)
        cf975['long'] = pandas.read_csv(os.path.join( summaries_dir, "cf"+str(M)+"Long975.csv") , sep=",", header=0, index_col=0)

        kfHatR['short'] = pandas.read_csv(os.path.join( summaries_dir, "kf"+str(M)+"ShortHatR.csv") , sep=",", header=0, index_col=0)
        kfHatR['control'] = pandas.read_csv(os.path.join( summaries_dir, "kf"+str(M)+"ControlHatR.csv") , sep=",", header=0, index_col=0)
        kfHatR['long'] = pandas.read_csv(os.path.join( summaries_dir, "kf"+str(M)+"LongHatR.csv") , sep=",", header=0, index_col=0)

        kfESS['short'] = pandas.read_csv(os.path.join( summaries_dir, "kf"+str(M)+"ShortESS.csv") , sep=",", header=0, index_col=0)
        kfESS['control'] = pandas.read_csv(os.path.join( summaries_dir, "kf"+str(M)+"ControlESS.csv") , sep=",", header=0, index_col=0)
        kfESS['long'] = pandas.read_csv(os.path.join( summaries_dir, "kf"+str(M)+"LongESS.csv") , sep=",", header=0, index_col=0)

        lpHatR['short'] = pandas.read_csv(os.path.join( summaries_dir, "lp"+str(M)+"ShortHatR.csv") , sep=",", header=0, index_col=0)
        lpHatR['control'] = pandas.read_csv(os.path.join( summaries_dir, "lp"+str(M)+"ControlHatR.csv") , sep=",", header=0, index_col=0)
        lpHatR['long'] = pandas.read_csv(os.path.join( summaries_dir, "lp"+str(M)+"LongHatR.csv") , sep=",", header=0, index_col=0)

        lpESS['short'] = pandas.read_csv(os.path.join( summaries_dir, "lp"+str(M)+"ShortESS.csv") , sep=",", header=0, index_col=0)
        lpESS['control'] = pandas.read_csv(os.path.join( summaries_dir, "lp"+str(M)+"ControlESS.csv") , sep=",", header=0, index_col=0)
        lpESS['long'] = pandas.read_csv(os.path.join( summaries_dir, "lp"+str(M)+"LongESS.csv") , sep=",", header=0, index_col=0)

        return  cfMean, cf25, cf975,   kfHatR, lpHatR, kfESS, lpESS


def order_cf(cFrames, summaries_order=None):
    """ Orders data frame rows and columns (used for instance for sorting one of two data frames for comparison) """

    cfMeanMain, cf25Main, cf975Main, kfHatRMain, lpHatRMain, kfESSMain, lpESSMain = cFrames

    if (summaries_order is None):

        warnstring = ">>> no ordering argument provided, returning data frame in original order"
        warnings.warn(warnstring, UserWarning)

        return cfMeanMain, cf25Main, cf975Main, kfHatRMain, lpHatRMain, kfESSMain, lpESSMain

    elif ( type(summaries_order) == list ):

        order = summaries_order

    elif ( type(summaries_order) == dict ):

        cfMeanOrder = summaries_order
        order = cfMeanOrder["control"].index

    elif ( type(summaries_order) == str ):
        cfMeanOrder, cf25Order, cf975Order, kfHatROrder, lpHatROrder, kfESSOrder, lpESSOrder = load_cf(summaries_order)

        order = cfMeanOrder["control"].index
    else:
        warnstring = ">>> unrecognized ordering argument, returning data frame in original order"
        warnings.warn(warnstring, UserWarning)

        order = cfMeanMain["control"].index

    cfMeanReorder = {}
    cfMeanReorder["short"] = cfMeanMain["short"].loc[order, order]
    cfMeanReorder["control"] = cfMeanMain["control"].loc[order, order]
    cfMeanReorder["long"] = cfMeanMain["long"].loc[order, order]

    cf25Reorder = {}
    cf25Reorder["short"] = cf25Main["short"].loc[order, order]
    cf25Reorder["control"] = cf25Main["control"].loc[order, order]
    cf25Reorder["long"] = cf25Main["long"].loc[order, order]

    cf975Reorder = {}
    cf975Reorder["short"] = cf975Main["short"].loc[order, order]
    cf975Reorder["control"] = cf975Main["control"].loc[order, order]
    cf975Reorder["long"] = cf975Main["long"].loc[order, order]

    kfHatRReorder = {}
    kfHatRReorder["short"] = kfHatRMain["short"].loc[order,  order]
    kfHatRReorder["control"] = kfHatRMain["control"].loc[order,  order]
    kfHatRReorder["long"] = kfHatRMain["long"].loc[order,  order]

    lpHatRReorder = {}
    lpHatRReorder["short"] = lpHatRMain["short"].loc[order, order]
    lpHatRReorder["control"] = lpHatRMain["control"].loc[order, order]
    lpHatRReorder["long"] = lpHatRMain["long"].loc[order, order]

    kfESSReorder = {}
    kfESSReorder["short"] = kfESSMain["short"].loc[order, order]
    kfESSReorder["control"] = kfESSMain["control"].loc[order, order]
    kfESSReorder["long"] = kfESSMain["long"].loc[order, order]

    lpESSReorder = {}
    lpESSReorder["short"] = lpESSMain["short"].loc[order, order]
    lpESSReorder["control"] = lpESSMain["control"].loc[order, order]
    lpESSReorder["long"] = lpESSMain["long"].loc[order, order]

    return cfMeanReorder, cf25Reorder, cf975Reorder, kfHatRReorder, lpHatRReorder, kfESSReorder, lpESSReorder


def cf_fillin(summaries_dir, order=None):


    cfMeanMain, cf25Main, cf975Main, kfHatRMain, lpHatRMain, kfESSMain, lpESSMain = load_cf(summaries_dir)
    cfMeanComplete = {}
    cf25Complete = {}
    cf975Complete = {}
    kfHatRComplete = {}
    lpHatRComplete = {}
    kfESSComplete = {}
    lpESSComplete = {}

    for key in cfMeanMain.keys():
        cfMeanArray, cf25Array, cf975Array, kfHatRArray, lpHatRArray, kfESSArray, lpESSArray = complete_sematary( (cfMeanMain[key].values, cf25Main[key].values, cf975Main[key].values, kfHatRMain[key].values, lpHatRMain[key].values, kfESSMain[key].values, lpESSMain[key].values) )

        cfMeanComplete[key] = pandas.DataFrame( cfMeanArray, index=cfMeanMain[key].index, columns=cfMeanMain[key].columns )
        cf25Complete[key] = pandas.DataFrame( cf25Array, index=cf25Main[key].index, columns=cf25Main[key].columns )
        cf975Complete[key] = pandas.DataFrame( cf975Array, index=cf975Main[key].index, columns=cf975Main[key].columns )
        kfHatRComplete[key] = pandas.DataFrame( kfHatRArray, index=cf25Main[key].index, columns=cf25Main[key].columns )
        lpHatRComplete[key] = pandas.DataFrame( lpHatRArray, index=cf975Main[key].index, columns=cf975Main[key].columns )
        kfESSComplete[key] = pandas.DataFrame( kfESSArray, index=cf25Main[key].index, columns=cf25Main[key].columns )
        lpESSComplete[key] = pandas.DataFrame( lpESSArray, index=cf975Main[key].index, columns=cf975Main[key].columns )

    cfMeanOrdered, cf25Ordered, cf975Ordered, kfHatROrdered, lpHatROrdered, kfESSOrdered, lpESSOrdered = order_cf((cfMeanMain, cf25Main, cf975Main, kfHatRMain, lpHatRMain, kfESSMain, lpESSMain), summaries_order=order)

    return cfMeanOrdered, cf25Ordered, cf975Ordered, kfHatROrdered, lpHatROrdered, kfESSOrdered, lpESSOrdered


@numba.jit
def complete_sematary(cfSummariesSelArray):
    """ fills triangular data frame empty side to output symmetric matrix """

    cfMeanSelComplete, cf25SelComplete, cf975SelComplete, kfHatRSelComplete, lpHatRSelComplete, kfESSSelComplete, lpESSSelComplete = cfSummariesSelArray

    for i in range( cfMeanSelComplete.shape[0] ):
        cfMeanSelComplete[i,i] = 1
        cf25SelComplete[i,i] = 1
        cf975SelComplete[i,i] = 1
        kfHatRSelComplete[i,i] = numpy.NaN
        lpHatRSelComplete[i,i] = numpy.NaN
        kfESSSelComplete[i,i] = numpy.NaN
        lpESSSelComplete[i,i] = numpy.NaN
        for j in range( i+1, cfMeanSelComplete.shape[0] ):
            cfMeanSelComplete[j,i] = cfMeanSelComplete[i,j]
            cf25SelComplete[j,i] = cf25SelComplete[i,j]
            cf975SelComplete[j,i] = cf975SelComplete[i,j]
            kfHatRSelComplete[j,i] = kfHatRSelComplete[i,j]
            lpHatRSelComplete[j,i] = lpHatRSelComplete[i,j]
            kfESSSelComplete[j,i] = kfESSSelComplete[i,j]
            lpESSSelComplete[j,i] = lpESSSelComplete[i,j]

    return cfMeanSelComplete, cf25SelComplete, cf975SelComplete, kfHatRSelComplete, lpHatRSelComplete, kfESSSelComplete, lpESSSelComplete


@numba.jit
def matrix_confidence_array(cfArray1, cfArray2):
    """ checks overlap of confidence intervals using arrays for two groups (1 and 2)"""

    # arguments must each pack an array with mean, lower condifence limit and upper confidence limit (assumed to be 95%, but will work with any limit, with different confidence)
    meanArray1, p25Array1, p975Array1 = cfArray1
    meanArray2, p25Array2, p975Array2 = cfArray2
    confidentArray = numpy.full( meanArray1.shape, numpy.NaN )

    for i in range( meanArray1.shape[0] ):
        confidentArray[i,i] = 1
        for j in range( i+1, meanArray1.shape[1] ):
            if ( (meanArray1[i,j] > meanArray2[i,j]) and (p25Array1[i,j] < p975Array2[i,j]) ):
                pass
            elif ( (meanArray1[i,j] < meanArray2[i,j]) and (p975Array1[i,j] > p25Array2[i,j]) ):
                pass
            else:
                confidentArray[i,j] = confidentArray[j,i] = meanArray1[i,j]

    return confidentArray


def matrix_confidence(treatmentFrames, controlFrames, relabelFrame=None):
    """ checks overlap of confidence intervals between two data frames """

    meanFrame1, p25Frame1, p975Frame1 = treatmentFrames
    meanFrame2, p25Frame2, p975Frame2 = controlFrames

    symbolist = relabelFrame.loc[meanFrame1.index].symbol if ( type(relabelFrame) != type(None) ) else meanFrame1.index

    if ( all(meanFrame1.index == meanFrame2.index) and all(meanFrame1.columns == meanFrame2.columns) ):
        confidentArray = matrix_confidence_array( (meanFrame1.values, p25Frame1.values, p975Frame1.values), (meanFrame2.values, p25Frame2.values, p975Frame2.values) )
        confidentFrame = pandas.DataFrame( confidentArray, index=symbolist, columns=symbolist )
    else:
        warnstring = ">>> index or columns of data frame do not match, returning nothing"
        warnings.warn(warnstring, UserWarning)
        confidentFrame = None

    return confidentFrame


def sex_overlap_array(selMale, selFemale):
    """ checks overlap between sexes """

    selOverlapArray = numpy.full( selMale.shape, numpy.NaN )
    for i in range( selMale.shape[0] ):
        selOverlapArray[i,i] = 1
        for j in range( i+1, selFemale.shape[0] ):
            if ( numpy.isnan(selMale[i,j]) or numpy.isnan(selFemale[i,j]) ):
                pass
            elif ( (selMale[i,j] > 0) and (selFemale[i,j] > 0) ):
                selOverlapArray[j,i] = 1
                selOverlapArray[i,j] = 1
            elif ( (selMale[i,j] < 0) and (selFemale[i,j] < 0) ):
                selOverlapArray[j,i] = -1
                selOverlapArray[i,j] = -1
            else:
                selOverlapArray[j,i] = 0
                selOverlapArray[i,j] = 0

    return selOverlapArray


def sex_overlap(selMale, selFemale):
    """ checks overlap between data frames for each sex """

    if ( all(selMale.index == selFemale.index) and all(selMale.columns == selFemale.columns) ):
        selOverlap = pandas.DataFrame( sex_overlap_array(selMale.values, selFemale.values), index=selMale.index, columns=selMale.columns )
    else:
        warnstring = ">>> index or columns of data frame do not match, returning nothing"
        warnings.warn(warnstring, UserWarning)
        selOverlap = None

    return selOverlap


def cf_kde(cfMatrix, fig=True):

    if fig==True:
        figX = figure(figsize=(10,8))

    cfMatrixShort = numpy.array( [ cfMatrix["short"].values[i,j] for i in range(1, cfMatrix["short"].shape[0]) for j in range(i+1,cfMatrix["short"].shape[1]) ] )

    cfMatrixControl = numpy.array( [ cfMatrix["control"].values[i,j] for i in range(1, cfMatrix["control"].shape[0]) for j in range(i+1,cfMatrix["control"].shape[1]) ] )

    cfMatrixLong = numpy.array( [ cfMatrix["long"].values[i,j] for i in range(1, cfMatrix["long"].shape[0]) for j in range(i+1,cfMatrix["long"].shape[1]) ] )


    seaborn.distplot(
                cfMatrixShort, hist=False, kde=True,
                bins=numpy.linspace(-1,1,20), color = 'DarkOrange',
                hist_kws={'edgecolor':'None'},
                kde_kws={'linewidth': 4},
                label="short"
                )

    seaborn.distplot(
                cfMatrixControl, hist=False, kde=True,
                bins=numpy.linspace(-1,1,20), color = 'Gray',
                hist_kws={'edgecolor':'None'},
                kde_kws={'linewidth': 4},
                label="control"
                )


    seaborn.distplot(
                cfMatrixLong, hist=False, kde=True,
                bins=numpy.linspace(-1,1,20), color = 'ForestGreen',
                hist_kws={'edgecolor':'None'},
                kde_kws={'linewidth': 4},
                label="long"
                )

    pyplot.xlim( [-1.2,1.2] )
    pyplot.xlabel( "correlation", fontsize=20 )
    pyplot.ylabel( "density", fontsize=20 )

    shortsum = numpy.round( numpy.nansum( numpy.abs( (cfMatrix["short"]-cfMatrix["control"]).values ), axis=(0,1) ), 2)
    longsum = numpy.round( numpy.nansum( numpy.abs( (cfMatrix["long"]-cfMatrix["control"]).values ), axis=(0,1) ), 2)

    shortstring = "+" + str(shortsum) if shortsum > 0 else str(shortsum)
    longstring = "+" + str(longsum) if shortsum > 0 else str(longsum)

    pyplot.title( "Aggregate interaction shifts (absolute) compared to control\n short: " + shortstring + ", long: " + longstring)

    return None


def interaction_frame(cfDict2, confidentFrames, hatRDict2={}, essDict2={}, verbo=True):

    cfMeanFem, cfMeanMale = cfDict2
    cfShortFem, cfControlFem, cfLongFem = cfMeanFem['short'], cfMeanFem['control'], cfMeanFem['long']
    cfShortMale, cfControlMale, cfLongMale = cfMeanMale['short'], cfMeanMale['control'], cfMeanMale['long']

    cfConfidentShortFem, cfConfidentShortMale, cfConfidentLongFem, cfConfidentLongMale = confidentFrames

    if ( len(hatRDict2) > 0 ):
        hatRFem, hatRMale = hatRDict2
        hatRShortFem, hatRControlFem, hatRLongFem = hatRFem['short'], hatRFem['control'], hatRFem['long']
        hatRShortMale, hatRControlMale, hatRLongMale = hatRMale['short'], hatRMale['control'], hatRMale['long']

    if ( len(essDict2) > 0 ):
        essFem, essMale = essDict2
        essShortFem, essControlFem, essLongFem = essFem['short'], essFem['control'], essFem['long']
        essShortMale, essControlMale, essLongMale = essMale['short'], essMale['control'], essMale['long']

    M = cfShortFem.shape[0]
    trilM = int( ( (M**2) - M )/2 )

    interactionFrame = pandas.DataFrame( index=range(trilM), columns = [ "fbid_one", "fbid_two", "gene_one", "gene_two", "female_short", "male_short", "female_control", "male_control", "female_long", "male_long", "significant_femshort", "significant_maleshort", "significant_femlong", "significant_malelong", "femshort_hatR", "maleshort_hatR", "femctrl_hatR", "malectrl_hatR", "femlong_hatR", "malelong_hatR", "femshort_ESS", "maleshort_ESS", "femctrl_ESS", "malectrl_ESS", "femlong_ESS", "malelong_ESS"] )

    sumi = 0
    for i,smbi in enumerate(cfShortFem.index):
        for j,smbj in enumerate(cfShortFem.columns):
            if (verbo==True):
                print(">>> row/column: " + str(i+1) + "/" + str(j+1) + ", fbid:", smbi, smbj)
            if (j>i):
                interactionFrame.loc[ sumi, "fbid_one" ] = cfShortFem.index[i]
                interactionFrame.loc[ sumi, "fbid_two" ] = cfShortFem.columns[j]

                interactionFrame.loc[ sumi, "gene_one" ] = cfConfidentShortFem.index[i]
                interactionFrame.loc[ sumi, "gene_two" ] = cfConfidentShortFem.columns[j]

                interactionFrame.loc[ sumi, "female_short" ] = cfShortFem.iloc[i, j]
                interactionFrame.loc[ sumi, "male_short" ] = cfShortMale.iloc[i, j]

                interactionFrame.loc[ sumi, "female_control" ] = cfControlFem.iloc[i, j]
                interactionFrame.loc[ sumi, "male_control" ] = cfControlMale.iloc[i, j]

                interactionFrame.loc[ sumi, "female_long" ] = cfLongFem.iloc[i, j]
                interactionFrame.loc[ sumi, "male_long" ] = cfLongMale.iloc[i, j]

                interactionFrame.loc[ sumi, "significant_femshort" ] = cfConfidentShortFem.iloc[i, j]
                interactionFrame.loc[ sumi, "significant_maleshort" ] = cfConfidentShortMale.iloc[i, j]

                interactionFrame.loc[ sumi, "significant_femlong" ] = cfConfidentLongFem.iloc[i, j]
                interactionFrame.loc[ sumi, "significant_malelong" ] = cfConfidentLongMale.iloc[i, j]

                #interactionFrame.loc[ sumi, "overlap_short" ] = overlapShort.iloc[i, j]
                #interactionFrame.loc[ sumi, "overlap_long" ] = overlapLong.iloc[i, j]

                if ( len(hatRDict2) > 0 ):

                    interactionFrame.loc[ sumi, "femshort_hatR" ] = hatRShortFem.iloc[i, j]
                    interactionFrame.loc[ sumi, "maleshort_hatR" ] = hatRShortMale.iloc[i, j]

                    interactionFrame.loc[ sumi, "femctrl_hatR" ] = hatRControlFem.iloc[i, j]
                    interactionFrame.loc[ sumi, "malectrl_hatR" ] = hatRControlMale.iloc[i, j]

                    interactionFrame.loc[ sumi, "femlong_hatR" ] = hatRLongFem.iloc[i, j]
                    interactionFrame.loc[ sumi, "malelong_hatR" ] = hatRLongMale.iloc[i, j]


                if ( len(essDict2) > 0 ):

                    interactionFrame.loc[ sumi, "femshort_ESS" ] = essShortFem.iloc[i, j]
                    interactionFrame.loc[ sumi, "maleshort_ESS" ] = essShortMale.iloc[i, j]

                    interactionFrame.loc[ sumi, "femctrl_ESS" ] = essControlFem.iloc[i, j]
                    interactionFrame.loc[ sumi, "malectrl_ESS" ] = essControlMale.iloc[i, j]

                    interactionFrame.loc[ sumi, "femlong_ESS" ] = essLongFem.iloc[i, j]
                    interactionFrame.loc[ sumi, "malelong_ESS" ] = essLongMale.iloc[i, j]

                sumi = sumi + 1

    return interactionFrame


def significant_interaction_frame(interactionFrame, deltaHatR=0.05, minESS=1000):

    significantInteractionFrame = interactionFrame[ ( interactionFrame.significant_femshort.notna() & (numpy.abs(1-interactionFrame.femshort_hatR)<deltaHatR) & (interactionFrame.femshort_ESS>minESS) & (numpy.abs(1-interactionFrame.femctrl_hatR)<deltaHatR) & (interactionFrame.femctrl_ESS>minESS) ) |  (interactionFrame.significant_maleshort.notna() & (numpy.abs(1-interactionFrame.maleshort_hatR)<deltaHatR) & (interactionFrame.maleshort_ESS>minESS) & (numpy.abs(1-interactionFrame.malectrl_hatR)<deltaHatR) & (interactionFrame.malectrl_ESS>minESS) ) | ( interactionFrame.significant_femlong.notna() & (numpy.abs(1-interactionFrame.femlong_hatR)<deltaHatR) & (interactionFrame.femlong_ESS>minESS) & (numpy.abs(1-interactionFrame.femctrl_hatR)<deltaHatR) & (interactionFrame.femctrl_ESS>minESS) ) | (interactionFrame.significant_malelong.notna() & (numpy.abs(1-interactionFrame.malelong_hatR)<deltaHatR) & (interactionFrame.malelong_ESS>minESS) & (numpy.abs(1-interactionFrame.malectrl_hatR)<deltaHatR) & (interactionFrame.malectrl_ESS>minESS)) ]

    return significantInteractionFrame


def significant_versus_notspearman(interactionFrame, spearmanFrame, pValue=0.05, significant=True, deltaHatR=0.05, minESS=1000):

    interactionFrame.index = [ interactionFrame.loc[id].fbid_one + interactionFrame.loc[id].fbid_two for id in interactionFrame.index ]

    None if all(interactionFrame.index==spearmanFrame.index) else warnings.warn(">>> no ordering argument provided, returning data frame in original order", UserWarning)

    versusFrame = pandas.concat( [interactionFrame, spearmanFrame[ ['corrFemShort', 'corrFemControl', 'corrFemLong', 'corrMaleShort', 'corrMaleControl', 'corrMaleLong', 'pFemShort', 'pFemControl', 'pFemLong', 'pMaleShort', 'pMaleControl', 'pMaleLong', 'bhFemShort', 'bhFemControl', 'bhFemLong', 'bhMaleShort', 'bhMaleControl', 'bhMaleLong'] ] ], axis=1 )

    significantNotSpearman = versusFrame[

    ( ( versusFrame.significant_femshort.notna() & (numpy.abs(1-versusFrame.femshort_hatR)<deltaHatR) & (versusFrame.femshort_ESS>minESS) & (numpy.abs(1-versusFrame.femctrl_hatR)<deltaHatR) & (versusFrame.femctrl_ESS>minESS) ) & (versusFrame.pFemShort>pValue) ) |

    ( ( versusFrame.significant_maleshort.notna() & (numpy.abs(1-versusFrame.maleshort_hatR)<deltaHatR) & (versusFrame.maleshort_ESS>minESS) & (numpy.abs(1-versusFrame.malectrl_hatR)<deltaHatR) & (versusFrame.malectrl_ESS>minESS) ) & (versusFrame.pMaleShort>pValue) ) |

    ( ( versusFrame.significant_femlong.notna() & (numpy.abs(1-versusFrame.femlong_hatR)<deltaHatR) & (versusFrame.femlong_ESS>minESS) & (numpy.abs(1-versusFrame.femctrl_hatR)<deltaHatR) & (versusFrame.femctrl_ESS>minESS) ) & (versusFrame.pFemLong>pValue) ) |

     ( (versusFrame.significant_malelong.notna() & (numpy.abs(1-versusFrame.malelong_hatR)<deltaHatR) & (versusFrame.malelong_ESS>minESS) & (numpy.abs(1-versusFrame.malectrl_hatR)<deltaHatR) & (versusFrame.malectrl_ESS>minESS) ) & (versusFrame.pMaleShort>pValue) ) ]


    if (significant==True):
        return significantNotSpearman
    else:
        return versusFrame


def significant_versus_notspearman2(interactionFrame, spearmanFrame, pValue=0.05, significant=True, deltaHatR=0.05, minESS=1000):

    interactionFrame.index = [ interactionFrame.loc[id].fbid_one + interactionFrame.loc[id].fbid_two for id in interactionFrame.index ]

    None if all(interactionFrame.index==spearmanFrame.index) else warnings.warn(">>> no ordering argument provided, returning data frame in original order", UserWarning)

    versusFrame = pandas.concat( [interactionFrame, spearmanFrame[ ['corrFemShort', 'corrFemControl', 'corrFemLong', 'corrMaleShort', 'corrMaleControl', 'corrMaleLong', 'pFemShort', 'pFemControl', 'pFemLong', 'pMaleShort', 'pMaleControl', 'pMaleLong', 'bhFemShort', 'bhFemControl', 'bhFemLong', 'bhMaleShort', 'bhMaleControl', 'bhMaleLong'] ] ], axis=1 )

    significantNotSpearman = versusFrame[

    ( ( versusFrame.significant_femshort.notna() & (numpy.abs(1-versusFrame.femshort_hatR)<deltaHatR) & (versusFrame.femshort_ESS>minESS) & (numpy.abs(1-versusFrame.femctrl_hatR)<deltaHatR) & (versusFrame.femctrl_ESS>minESS) ) & ( ( (versusFrame.pFemShort>pValue) & (versusFrame.pFemControl>pValue) ) | ( (versusFrame.pFemShort<pValue) & (versusFrame.pFemControl<pValue) ) ) ) |

    ( ( versusFrame.significant_maleshort.notna() & (numpy.abs(1-versusFrame.maleshort_hatR)<deltaHatR) & (versusFrame.maleshort_ESS>minESS) & (numpy.abs(1-versusFrame.malectrl_hatR)<deltaHatR) & (versusFrame.malectrl_ESS>minESS) ) & ( ( (versusFrame.pMaleShort>pValue) & (versusFrame.pMaleControl>pValue) ) | ( (versusFrame.pMaleShort<pValue) & (versusFrame.pMaleControl<pValue) ) ) ) |

    ( ( versusFrame.significant_femlong.notna() & (numpy.abs(1-versusFrame.femlong_hatR)<deltaHatR) & (versusFrame.femlong_ESS>minESS) & (numpy.abs(1-versusFrame.femctrl_hatR)<deltaHatR) & (versusFrame.femctrl_ESS>minESS) ) & ( ( (versusFrame.pFemLong>pValue) & (versusFrame.pFemControl>pValue) ) | ( (versusFrame.pFemLong<pValue) & (versusFrame.pFemControl<pValue) ) ) ) |

     ( (versusFrame.significant_malelong.notna() & (numpy.abs(1-versusFrame.malelong_hatR)<deltaHatR) & (versusFrame.malelong_ESS>minESS) & (numpy.abs(1-versusFrame.malectrl_hatR)<deltaHatR) & (versusFrame.malectrl_ESS>minESS) ) & ( ( (versusFrame.pMaleShort>pValue) & (versusFrame.pMaleControl>pValue) ) | ( (versusFrame.pMaleShort<pValue) & (versusFrame.pMaleControl<pValue) ) ) ) ]


    if (significant==True):
        return significantNotSpearman
    else:
        return versusFrame


def overlap_versus_notspearman(interactionFrame, spearmanFrame, significant=True, deltaHatR=0.05, minESS=1000):

    interactionFrame.index = [ interactionFrame.loc[id].fbid_one + interactionFrame.loc[id].fbid_two for id in interactionFrame.index ]

    None if all(interactionFrame.index==spearmanFrame.index) else warnings.warn(">>> no ordering argument provided, returning data frame in original order", UserWarning)

    versusFrame = pandas.concat( [interactionFrame, spearmanFrame[ ['corrFemShort', 'corrFemControl', 'corrFemLong', 'corrMaleShort', 'corrMaleControl', 'corrMaleLong', 'pFemShort', 'pFemControl', 'pFemLong', 'pMaleShort', 'pMaleControl', 'pMaleLong', 'bhFemShort', 'bhFemControl', 'bhFemLong', 'bhMaleShort', 'bhMaleControl', 'bhMaleLong', 'lowerFemShort', 'upperFemShort', 'lowerFemControl', 'upperFemControl', 'lowerFemLong', 'upperFemLong', 'lowerMaleShort', 'upperMaleShort', 'lowerMaleControl', 'upperMaleControl', 'lowerMaleLong', 'upperMaleLong'] ] ], axis=1 )

    significantNotSpearman = versusFrame[
    (
    ( versusFrame.significant_femshort.notna() & (numpy.abs(1-versusFrame.femshort_hatR)<deltaHatR) & (versusFrame.femshort_ESS>minESS) & (numpy.abs(1-versusFrame.femctrl_hatR)<deltaHatR) & (versusFrame.femctrl_ESS>minESS)
    ) & ~(
    ( (versusFrame['corrFemControl']<versusFrame['corrFemShort']) & (versusFrame['upperFemControl']<versusFrame['lowerFemShort']) ) | ( (versusFrame['corrFemControl']>versusFrame['corrFemShort']) & (versusFrame['lowerFemControl']>versusFrame['upperFemShort']) ) )
    ) | (

    ( versusFrame.significant_maleshort.notna() & (numpy.abs(1-versusFrame.maleshort_hatR)<deltaHatR) & (versusFrame.maleshort_ESS>minESS) & (numpy.abs(1-versusFrame.malectrl_hatR)<deltaHatR) & (versusFrame.malectrl_ESS>minESS) ) & ~(

    ( (versusFrame['corrMaleControl']<versusFrame['corrMaleShort']) & (versusFrame['upperMaleControl']<versusFrame['lowerMaleShort']) ) | ( (versusFrame['corrMaleControl']>versusFrame['corrMaleShort']) & (versusFrame['lowerMaleControl']>versusFrame['upperMaleShort']) ) )
    ) | (

    ( versusFrame.significant_femlong.notna() & (numpy.abs(1-versusFrame.femlong_hatR)<deltaHatR) & (versusFrame.femlong_ESS>minESS) & (numpy.abs(1-versusFrame.femctrl_hatR)<deltaHatR) & (versusFrame.femctrl_ESS>minESS) ) & ~(

    ( (versusFrame['corrFemControl']<versusFrame['corrFemLong']) & (versusFrame['upperFemControl']<versusFrame['lowerFemLong']) ) | ( (versusFrame['corrFemControl']>versusFrame['corrFemLong']) & (versusFrame['lowerFemControl']>versusFrame['upperFemLong']) ) )
    ) | (
    (versusFrame.significant_malelong.notna() & (numpy.abs(1-versusFrame.malelong_hatR)<deltaHatR) & (versusFrame.malelong_ESS>minESS) & (numpy.abs(1-versusFrame.malectrl_hatR)<deltaHatR) & (versusFrame.malectrl_ESS>minESS) ) & ~(

    ( (versusFrame['corrMaleControl']<versusFrame['corrMaleLong']) & (versusFrame['upperMaleControl']<versusFrame['lowerMaleLong']) ) | ( (versusFrame['corrMaleControl']>versusFrame['corrMaleLong']) & (versusFrame['lowerMaleControl']>versusFrame['upperMaleLong']) ) ) )
    ]

    if (significant==True):
        return significantNotSpearman
    else:
        return versusFrame
