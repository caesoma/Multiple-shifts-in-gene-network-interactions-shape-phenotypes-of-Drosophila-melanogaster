#/usr/bin/python3
import os
import numpy
import pandas
import scipy.stats

import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import figure, plot, show

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


def plot_sel_panel(sell, metadata, fbidata, fullMean, full25, full975, reducedMean, reduced25, reduced975, colors=["Crimson", "CornFlowerBlue"]):

    generation = metadata.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].generation.values
    sizeFactors1 = metadata.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].sizeFactors.values
    sizeFactors2 = metadata.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].sizeFactors.values

    rep1data = fbidata.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values / sizeFactors1
    rep2data = fbidata.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values / sizeFactors2

    rep1FullMean = fullMean.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values
    rep2FullMean = fullMean.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values

    rep1Full25 = full25.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values
    rep2Full25 = full25.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values

    rep1Full975 = full975.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values
    rep2Full975 = full975.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values

    rep1ReducedMean = reducedMean.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values
    rep2ReducedMean = reducedMean.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values

    rep1Reduced25 = reduced25.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values
    rep2Reduced25 = reduced25.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values

    rep1Reduced975 = reduced975.loc[ ( (metadata.sel==sell) & (metadata.Rep==1) ) ].values
    rep2Reduced975 = reduced975.loc[ ( (metadata.sel==sell) & (metadata.Rep==2) ) ].values

    color1, color2 = colors

    plot(generation, rep1data, 'o', color='0.4', mec=None, label="Rep 1")
    plot(generation, rep2data, 'o', color='0.7', mec=None, label="Rep 2")

    pyplot.fill_between(generation, rep1Reduced25, rep1Reduced975, color=color2, linewidth=0, alpha=0.2)
    pyplot.fill_between(generation, rep2Reduced25, rep2Reduced975, color=color2, linewidth=0, alpha=0.2)

    plot(generation, rep1ReducedMean, ":", linewidth=2, color=color2, label="Reduced")
    plot(generation, rep2ReducedMean, ":", linewidth=2, color=color2, alpha=0.7)

    pyplot.fill_between(generation, rep2Full25, rep2Full975, color=color1, linewidth=0, alpha=0.2)
    pyplot.fill_between(generation, rep1Full25, rep1Full975, color=color1, linewidth=0, alpha=0.2)


    plot(generation, rep1FullMean, linewidth=2, color=color1, label="Full")
    plot(generation, rep2FullMean, linewidth=2, color=color1, alpha=0.7)

    return None


def hglm_plot(fbid, sex, metadata, data, results, fullMean, full25, full975, reducedMean, reduced25, reduced975, symbolFrame=pandas.DataFrame([]), colors=["DarkOrange", "MediumAquamarine"], labels=True, savepath=""):

    symbol = symbolFrame.loc[fbid].symbol if (symbolFrame.shape[0]>0) else fbid

    panel = lambda i,j : pyplot.subplot2grid( shape=(1,3), loc=(i,j) )

    figj = figure( figsize=(15,5) )

    panel(0,0)
    plot_sel_panel("short", metadata, data[fbid], fullMean[fbid], full25[fbid], full975[fbid], reducedMean[fbid], reduced25[fbid], reduced975[fbid], colors=colors)

    pyplot.ylim( [ 0.9 * min(data[fbid] / metadata.sizeFactors), 1.1 * max( data[fbid] / metadata.sizeFactors) ] )
    pyplot.title(symbol + "\nshort", fontsize=16)
    pyplot.ylabel( "Normalized \ngene expresion", fontsize=16)

    panel(0,1)
    plot_sel_panel("control", metadata, data[fbid], fullMean[fbid], full25[fbid], full975[fbid], reducedMean[fbid], reduced25[fbid], reduced975[fbid], colors=colors)

    pyplot.yticks([])
    pyplot.ylim( [ 0.9 * min(data[fbid] / metadata.sizeFactors), 1.1 * max( data[fbid] / metadata.sizeFactors) ] )

    pyplot.xlabel( "Generation", fontsize=16)
    pvalue = numpy.round(results.loc[fbid].p, 4) if numpy.isnan(results.loc[fbid].padj) else numpy.round(results.loc[fbid].padj, 4)
    pstring = "$p=$" + str(pvalue) if (pvalue>0) else "$p<0.001$"
    pyplot.title( pstring + "\n control", fontsize=16)
    pyplot.legend() if (labels==True) else None

    panel(0,2)
    plot_sel_panel("long", metadata, data[fbid], fullMean[fbid], full25[fbid], full975[fbid], reducedMean[fbid], reduced25[fbid], reduced975[fbid], colors=colors)

    pyplot.yticks([])
    pyplot.ylim( [ 0.9 * min(data[fbid] / metadata.sizeFactors), 1.1 * max( data[fbid] / metadata.sizeFactors) ] )
    pyplot.title( "long", fontsize=16)

    if (savepath != ""):
        figj.savefig( savepath )
    else:
        print(">>> figure not saved")

    return None
