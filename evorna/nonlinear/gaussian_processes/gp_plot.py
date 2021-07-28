#!/usr/bin/python3

"""
Functions to compute correlations between channels form Gaussian Process
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "3.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import os, sys, platform
import numpy, numba
import pandas
import networkx

import matplotlib, seaborn
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import figure, plot, show

import warnings


matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

def gp_scheme_plot(symbol, fbidata, metadata, gPosterior1, gPosterior2, selscheme, gcolor):
    """ plot one figure panel with single-channel Gaussian Process """
    # define axes for data and model
    poolGenerations = metadata[ (metadata.sel==selscheme) & (metadata.Rep==1) ].generation.values
    continuousGen = numpy.arange(poolGenerations[0], poolGenerations[-1]+1.1, 0.1)

    # get size factors
    sizeFactors1 = metadata[ ( metadata.sel==selscheme ) & ( metadata.Rep==1 ) ].sizeFactors.values
    sizeFactors2 = metadata[ ( metadata.sel==selscheme ) & ( metadata.Rep==2 ) ].sizeFactors.values

    # define normalized data and model expected values and CIs
    y1 = fbidata[ ( metadata.sel == selscheme ) & ( metadata.Rep==1 ) ]
    y2 = fbidata[ ( metadata.sel==selscheme ) & ( metadata.Rep==2 ) ]

    gp1mean = numpy.exp( numpy.mean( gPosterior1, axis=(0,1) ) )
    gp1p25  = numpy.exp( numpy.percentile( gPosterior1, 2.5, axis=(0,1) ) )
    gp1p975 = numpy.exp( numpy.percentile( gPosterior1, 97.5, axis=(0,1) ) )

    gp2mean = numpy.exp( numpy.mean( gPosterior2, axis=(0,1) ) )
    gp2p25  = numpy.exp( numpy.percentile( gPosterior2, 2.5, axis=(0,1) ) )
    gp2p975 = numpy.exp( numpy.percentile( gPosterior2, 97.5, axis=(0,1) ) )

    plot(poolGenerations, y1/sizeFactors1, 'o', color='0.4', mec=None)
    plot(poolGenerations, y2/sizeFactors2, 'o', color='0.7', mec=None)

    # pyplot.axhline( numpy.mean(y1+y2), linestyle='--', color=gcolor);

    plot(continuousGen, gp1mean, linewidth=2, color=gcolor);
    pyplot.fill_between(continuousGen, gp1p25, gp1p975, color=gcolor, linewidth=0, alpha=0.2);

    plot(continuousGen, gp2mean, linewidth=2, color=gcolor, alpha=0.8);
    pyplot.fill_between(continuousGen, gp2p25, gp2p975, color=gcolor, linewidth=0, alpha=0.2);

    pyplot.xticks(poolGenerations[::4])
    pyplot.ylim([0.9*min(fbidata.values / metadata.sizeFactors.values), 1.1*max(fbidata.values / metadata.sizeFactors.values)])
    pyplot.title(selscheme)

    return None


def gp_manuscript_single_plot(save_path, symbol, fbidata, metadata, gPosterior, color="CornFlowerBlue"):
    """ plot entire figure with single-channel Gaussian Process """

    gPosterior1s = gPosterior['shortRep1']
    gPosterior2s = gPosterior['shortRep2']
    gPosterior1c = gPosterior['controlRep1']
    gPosterior2c = gPosterior['controlRep2']
    gPosterior1l = gPosterior['longRep1']
    gPosterior2l = gPosterior['longRep2']

    figi = figure(figsize=(13, 5))

    panel = lambda i,j: pyplot.subplot2grid( shape=(1,3), loc=(i,j) )

    ax = panel(0,0)
    gp_scheme_plot(symbol, fbidata, metadata, gPosterior1s, gPosterior2s, "short", color)
    pyplot.text(0.2, 0.9, "$"+symbol+"$", fontsize=16, color=color, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    pyplot.ylabel("Normalized\n gene expression", fontsize=16)

    ax = panel(0,1)
    gp_scheme_plot(symbol, fbidata, metadata, gPosterior1c, gPosterior2c, "control", color)
    pyplot.xlabel("Generation", fontsize=16)

    ax = panel(0,2)
    gp_scheme_plot(symbol, fbidata, metadata, gPosterior1l, gPosterior2l, "long", color)

    if (save_path != "") :
        figi.savefig(save_path)
    else:
        print(">>> figure not saved")

    return None


def multigp_scheme_plot(symbol, fbidata, metadata, gPosterior1, gPosterior2, selscheme, gcolor, xtic=True, ytic=True, figtitle=True):
    """ plot one figure panel with dual-channel Gaussian Process """

    # define axes
    poolGenerations = metadata[(metadata.sel==selscheme) & (metadata.Rep==1)].generation.values
    continuousGen = numpy.arange(poolGenerations[0], poolGenerations[-1]+1.1, 0.1)

    # get data and normalizing factors
    y1 = fbidata[ ( metadata.sel == selscheme ) & ( metadata.Rep==1 ) ]
    y2 = fbidata[ ( metadata.sel==selscheme ) & ( metadata.Rep==2 ) ]

    sizeFactors1 = metadata[ ( metadata.sel==selscheme ) & ( metadata.Rep==1 ) ].sizeFactors.values
    sizeFactors2 = metadata[ ( metadata.sel==selscheme ) & ( metadata.Rep==2 ) ].sizeFactors.values

    # compute summaries from GP posterior distribution
    gp1mean = numpy.exp( numpy.mean( gPosterior1, axis=(0,1) ) )
    gp1p25  = numpy.exp( numpy.percentile( gPosterior1, 2.5, axis=(0,1) ) )
    gp1p975 = numpy.exp( numpy.percentile( gPosterior1, 97.5, axis=(0,1) ) )

    gp2mean = numpy.exp( numpy.mean( gPosterior2, axis=(0,1) ) )
    gp2p25  = numpy.exp( numpy.percentile( gPosterior2, 2.5, axis=(0,1) ) )
    gp2p975 = numpy.exp( numpy.percentile( gPosterior2, 97.5, axis=(0,1) ) )

    # plot all above together
    plot(poolGenerations, y1/sizeFactors1, 'o', color='0.4', mec=None)
    plot(poolGenerations, y2/sizeFactors2, 'o', color='0.7', mec=None)

    # pyplot.axhline( numpy.exp( meangp ), linestyle='--', color=gcolor);

    plot(continuousGen, gp1mean, linewidth=2, color=gcolor);
    pyplot.fill_between(continuousGen, gp1p25, gp1p975, color=gcolor, linewidth=0, alpha=0.2);

    plot(continuousGen, gp2mean, linewidth=2, color=gcolor, alpha=0.8);
    pyplot.fill_between(continuousGen, gp2p25, gp2p975, color=gcolor, linewidth=0, alpha=0.2);

    pyplot.xticks(poolGenerations[::4]) if (xtic == True) else pyplot.xticks([])
    pyplot.yticks([]) if (ytic == False) else None

    pyplot.ylim([0.9*min(fbidata.values / metadata.sizeFactors.values), 1.1*max(fbidata.values / metadata.sizeFactors.values)])
    pyplot.title(selscheme, fontsize=16) if (figtitle == True) else None

    pyplot.ylim([0.9*min(fbidata.values / metadata.sizeFactors.values), 1.1*max(fbidata.values / metadata.sizeFactors.values)])

    return None


def gp_manuscript_pair_plot(save_path, symbolist, fbidata2, metadata, gPosterior, rho={}, significance=["",""], color=["RoyalBlue", "Crimson"]):
    """ plot entire figure with dual-channel Gaussian Process (as seen in Souto-Maior et al. 2021)"""

    gPosterior1s = gPosterior['shortRep1']
    gPosterior2s = gPosterior['shortRep2']
    gPosterior1c = gPosterior['controlRep1']
    gPosterior2c = gPosterior['controlRep2']
    gPosterior1l = gPosterior['longRep1']
    gPosterior2l = gPosterior['longRep2']

    symbolPair = list( symbolist.loc[fbidata2.columns] )

    figij = figure( figsize=(12, 8) )
    panel = lambda i,j: pyplot.subplot2grid( shape=(2,3), loc=(i,j) )

    for m, fbid in enumerate( list( fbidata2.columns ) ):
        if (m==0):
            xtic=False
            figtitle=True
        elif (m==1):
            xtic=True
            figtitle=False

        ax = panel(m, 0)
        multigp_scheme_plot(symbolist[fbid], fbidata2[fbid], metadata, gPosterior1s[:,:,:,m], gPosterior2s[:,:,:,m], "short", gcolor=color[m], xtic=xtic, ytic=True, figtitle=figtitle)
        pyplot.title( "$\\rho_s = $" + str( numpy.round(rho['short'], 2) ) + significance[0] , fontsize=16 ) if ( (m==1) and (rho.keys() != {}.keys()) ) else None
        pyplot.text(0.2, 0.9, "$"+symbolist[fbid]+"$", fontsize=16, color=color[m], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        pyplot.ylabel(50*" "+"Normalized gene expression", fontsize=16) if (m==1) else None


        ax = panel(m, 1)
        multigp_scheme_plot(symbolist[fbid], fbidata2[fbid], metadata, gPosterior1c[:,:,:,m], gPosterior2c[:,:,:,m], "control", gcolor=color[m], xtic=xtic, ytic=False, figtitle=figtitle)
        pyplot.title( "$\\rho_c = $" + str( numpy.round(rho['control'], 2) ), fontsize=16 ) if ( (m==1) and (rho.keys() != {}.keys()) ) else None
        pyplot.xlabel("Generation", fontsize=16) if (m==1) else None

        ax = panel(m, 2)
        multigp_scheme_plot(symbolist[fbid], fbidata2[fbid], metadata, gPosterior1l[:,:,:,m], gPosterior2l[:,:,:,m], "long", gcolor=color[m], xtic=xtic, ytic=False, figtitle=figtitle)
        pyplot.title( "$\\rho_l = $" + str( numpy.round(rho['long'], 2) ) + significance[1], fontsize=16 ) if ( (m==1) and (rho.keys() != {}.keys()) ) else None

    if (save_path != ""):
        figij.savefig(save_path)
    else:
        print(">>> figure not saved")

    return None


def plot_Cf_ij(dictraces, fbidij=["",""], burn=0, verbo=False):
    """ plots posterior distributions of correlation parameters (these figures were not used in publications)"""

    fbidi, fbidj = fbidij
    print(">>> genes:", fbidi, "/", fbidj) if (verbo==True) else None

    dictraceShort, dictraceControl, dictraceLong = dictraces

    # compute posteriors for correlations in Multi-Channel Gaussian Process
    cfShortij = dictraceShort['kfTril'][burn:,:,0] / numpy.sqrt(dictraceShort['kfDiag'][burn:,:,0] * dictraceShort['kfDiag'][burn:,:,1] )
    cfControlij = dictraceControl['kfTril'][burn:,:,0] / numpy.sqrt( dictraceControl['kfDiag'][burn:,:,0] * dictraceControl['kfDiag'][burn:,:,1] )
    cfLongij = dictraceLong['kfTril'][burn:,:,0] / numpy.sqrt(dictraceLong['kfDiag'][burn:,:,0] * dictraceLong['kfDiag'][burn:,:,1] )

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

    print(">>> plotting distributions:") if (verbo==True) else None

    pyplot.figure(figsize=(12,10))
    pyplot.axvline(cfShort25ij, linestyle='--', color="DarkOrange", alpha=0.7)
    pyplot.axvline(cfShort975ij, linestyle='--', color="DarkOrange", alpha=0.7)

    pyplot.axvline(cfControl25ij, linestyle='--', color="Black", alpha=0.7)
    pyplot.axvline(cfControl975ij, linestyle='--', color="Black", alpha=0.7)

    pyplot.axvline(cfLong25ij, linestyle='--', color="ForestGreen", alpha=0.7)
    pyplot.axvline(cfLong975ij, linestyle='--', color="ForestGreen", alpha=0.7)

    pyplot.hist(cfShortij.reshape([-1,1])[:,0], bins=50, color="DarkOrange", alpha=0.7, label="short")
    pyplot.hist(cfControlij.reshape([-1,1])[:,0], bins=50, color="Black", alpha=0.7, label="control")
    pyplot.hist(cfLongij.reshape([-1,1])[:,0], bins=50, color="ForestGreen", alpha=0.7, label="long")

    pyplot.legend()

    return cfMeanij, cf25ij, cf975ij


def plotK_panel(summaryK, lower, upper, xt=False, yt=False, shift=0, bar=False):
    """ plot one panel with heatmap from GP correlations """
    M = summaryK.shape[0]

    pyplot.imshow(summaryK, cmap=matplotlib.cm.Spectral, vmin=lower, vmax=upper, norm=None)  # viridis, Spectral, inferno
    if bar: pyplot.colorbar(shrink=0.88)

    if ( (xt==True) and ( (M > 50)) ):
            exthics = [ sym for i,sym in enumerate( list(summaryK.columns) ) if i in range(shift,M,2) ]
            pyplot.xticks( range(shift, M, 2), exthics, rotation='vertical', fontsize=8)
    elif (xt==True):
        pyplot.xticks( range(M), list(summaryK.columns), rotation='vertical', fontsize=5.5)
    else:
        pyplot.xticks([])

    if (yt==True):
        pyplot.yticks(range(M), list(summaryK.index), fontsize=6)
    else:
        pyplot.yticks([])

    return None


def plotK(channelFrame, filepath="", correlation=False, symbolist=pandas.DataFrame([])):
    """ plot all three panels with heatmap from GP correlations (basic structure for figure 4 in Souto-Maior et al. 2021, except a 4th panel is included so the color bar doesn't distort the rest of the figure, and requires cuting that out to obtain the final figure) """

    if (symbolist.shape[0]==0):
        shortFrame = channelFrame['short']
        controlFrame = channelFrame['control']
        longFrame = channelFrame['long']
    else:
        shortFrame = pandas.DataFrame( channelFrame['short'].values, index=symbolist.loc[ channelFrame['short'].index ], columns=symbolist.loc[ channelFrame['short'].index ] )
        controlFrame = pandas.DataFrame( channelFrame['control'].values, index=symbolist.loc[ channelFrame['control'].index ], columns=symbolist.loc[ channelFrame['control'].index ] )
        longFrame = pandas.DataFrame( channelFrame['long'].values, index=symbolist.loc[ channelFrame['long'].index ], columns=symbolist.loc[ channelFrame['long'].index ] )

    maxk = 1 if (correlation==True) else numpy.nanmax( pandas.concat( [shortFrame, controlFrame, longFrame] ).values, axis=(0,1) )
    mink = -1 if (correlation==True) else numpy.nanmin( pandas.concat( [shortFrame, controlFrame, longFrame] ).values, axis=(0,1) )

    M = controlFrame.shape[0]

    figureK = figure( figsize = (40, 10) );

    pyplot.subplot(1, 4, 1)
    plotK_panel(shortFrame, mink, maxk, xt=True, yt=True, bar=False)
    pyplot.title("short", fontsize=24)

    pyplot.subplot(1, 4, 2)
    plotK_panel(controlFrame, mink, maxk, xt=True, yt=False, shift=1, bar=False)
    pyplot.title("control", fontsize=24)

    pyplot.subplot(1, 4, 3)
    plotK_panel(longFrame, mink, maxk, xt=True, yt=False)
    pyplot.title("long", fontsize=24)

    pyplot.subplot(1, 4, 4)
    pyplot.imshow(pandas.DataFrame(numpy.zeros([M,M])), cmap=matplotlib.cm.Spectral, vmin=mink, vmax=maxk, norm=None)
    pyplot.colorbar(shrink=0.88) #, labelsize=20)  # fraction=0.05, pad=0.04)  #

    if (filepath!=""):
        print(">>> saving figure...")
        figureK.savefig(filepath, bbox_inches="tight")

    return None


def plotK_significant( confidentFrames, filepath="", correlation=True):
    """ plot correlations different from contorls in four sex-selection data frames (heatmap version of figure 5, which is not a Python plot) """

    confidentShortM, confidentLongM, confidentShortF, confidentLongF = confidentFrames

    maxk = 1 if (correlation==True) else numpy.nanmax( pandas.concat( [ confidentShortM, confidentLongM, confidentShortF, confidentLongF ] ).values, axis=(0,1) )
    mink = -1 if (correlation==True) else numpy.nanmin( pandas.concat( [ confidentShortM, confidentLongM, confidentShortF, confidentLongF ] ).values, axis=(0,1) )

    M = confidentShortM.shape[0]

    subplotdim = (2,3)
    panel = lambda panx, pany: pyplot.subplot2grid( subplotdim, (panx, pany) )

    figureK = figure( figsize = (24, 20) );

    panel(0,0)
    plotK(confidentShortM, mink, maxk, xt=False, yt=True, bar=False)
    pyplot.title("Males, Short")

    panel(0,1)
    plotK(confidentShortF, mink, maxk, xt=False, yt=False, shift=1, bar=False)
    pyplot.title("Females, Short")

    panel(1,0)
    plotK(confidentLongM, mink, maxk, xt=True, yt=True, bar=False)
    pyplot.title("Males, Long")

    panel(1,1)
    plotK(confidentLongF, mink, maxk, xt=True, yt=False, bar=False)
    pyplot.title("Females, Long")

    ax = pyplot.subplot2grid( subplotdim, (0, 2), rowspan=2)
    ax.axis('off')
    pyplot.colorbar(shrink=0.9)  # , fraction=0.05, pad=0.04)


    if (filepath!=""):
        print(">>> saving figure...")
        figureK.savefig(filepath, bbox_inches="tight")

    return None



def plotK_overlap( confidentFrames, filepath=""):  #, correlation=True):
    """ plot overlap between correlation frames different from controls """

    confidentShort, confidentLong = confidentFrames

    subplotdim = (1,3)
    panel = lambda panx, pany: pyplot.subplot2grid( subplotdim, (panx, pany) )

    figureK = figure( figsize = (30, 10) );

    panel(0,0)
    plotK(confidentShort, -1, 1, xt=True, yt=False, bar=False)
    pyplot.title("Short")

    panel(0,1)
    plotK(confidentLong, -1, 1, xt=True, yt=True, bar=False)
    pyplot.title("Long")

    panel(0,2)
    pyplot.imshow( pandas.DataFrame( numpy.zeros( confidentShort.shape ) ), cmap=matplotlib.cm.Spectral, vmin=-1, vmax=1, norm=None)
    pyplot.colorbar(shrink=0.88)

    if (filepath!=""):
        print(">>> saving figure...")
        figureK.savefig(filepath, bbox_inches="tight")

    return None


def cf_kde(cfMatrix, fig=True):
    """ Plots aggregate shifts in interactions compared to control """

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


def graph(cFrame, cutoff=0, colormap=matplotlib.cm.Spectral):
    """ Crude graph visualization from correlation frame """
    G = networkx.Graph()

    colorlist = numpy.linspace(-1, 1, 201)
    colorray = colormap(colorlist)

    edgelist = []
    correlations = []
    colorlations = []

    for i,smbi in enumerate(cFrame.index):
        cFrame.loc[smbi, smbi] = numpy.NaN
        if all( numpy.isnan( cFrame.iloc[i,:] ) ):
            pass
        else:
            G.add_node(smbi)
        for j,smbj in enumerate(cFrame.index):
            if numpy.isnan(cFrame.loc[smbi, smbj]):
                pass
            elif ( numpy.abs(cFrame.loc[smbi, smbj]) < cutoff):
                pass
            elif (j>i):
                G.add_edge(smbi, smbj)
                edgelist.append((smbi, smbj))
                correlations.append( cFrame.loc[smbi, smbj] )
                colorlations.append( colorray[ int( 100*(numpy.round(cFrame.loc[smbi, smbj], 2) ) + 1)  ] )

    figX = figure(figsize=(8,8))

    networkx.draw(G, with_labels=True, node_size=75, node_color='CornFlowerBlue', edge_color=colorlations, width=0.1/numpy.abs(correlations))  #, width=[1,2,1,2,1], linewidths=2);

    #pyplot.xlim(-0.05,1.05)
    #pyplot.ylim(-0.05,1.05)
    #pyplot.axis('off')

    return None
