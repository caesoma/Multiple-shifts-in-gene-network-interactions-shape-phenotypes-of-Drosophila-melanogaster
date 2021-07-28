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

import scipy.stats
import matplotlib
import pandas

from matplotlib.pyplot import figure, plot, show
import matplotlib.pyplot as pyplot

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


def sleep_panel(traitSummary, trait, sub=False, title=False):
    """ plot single panel with trait summary over generations """

    generation = list( set( traitSummary.Generation ) )

    figure() if (sub==False) else None

    pyplot.title(trait, fontsize=16) if (title==True) else None

    plot(generation, traitSummary.loc[((traitSummary.Sel == "short") & (traitSummary.Rep == 1)), trait], color="DarkOrange", marker='s')
    plot(generation, traitSummary.loc[((traitSummary.Sel == "short") & (traitSummary.Rep == 2)), trait], color="Red", marker='s')
    plot(generation, traitSummary.loc[((traitSummary.Sel == "control") & (traitSummary.Rep == 1)), trait], color='0.7', marker='o')
    plot(generation, traitSummary.loc[((traitSummary.Sel == "control") & (traitSummary.Rep == 2)), trait], color='0', marker='o')
    plot(generation, traitSummary.loc[((traitSummary.Sel == "long") & (traitSummary.Rep == 1)), trait], color="LightGreen", marker='^')
    plot(generation, traitSummary.loc[((traitSummary.Sel == "long") & (traitSummary.Rep == 2)), trait], color="ForestGreen", marker='^')

    pyplot.xlabel("Generation", fontsize=16) if (sub==False) else None
    pyplot.xticks( [ int(generation[i]) for i in range(0, len(generation), 2 ) ] )

    return None


def trait_list(traitSummary, traitlist):
    """ plot panels as separate figures """
    if type(traitlist) == str:
        newlist = [traitlist]
    elif type(traitlist) == list:
        newlist = traitlist
    else:
        raise( ValueError("argument is not a string or list of strings") )

    [ sleep_panel(traitSummary, sub=False, trait=tr, title=True) for tr in newlist ]

    return None


def one_trait(traitSummary1, traitSummary2, tr, yaxlabels=["Mean","$CV_E$"], title=["", ""], savepath=""):
    """ plot pair of panels with one summary each """

    panel = lambda panx, pany: pyplot.subplot2grid( (1,2), (panx, pany) )

    fig1AB = figure( figsize = (18, 7) )
    title1, title2 = title
    yaxlabel1, yaxlabel2 = yaxlabels
    ax = panel(0, 0)
    sleep_panel(traitSummary1, trait=tr, sub=True, title=False)
    pyplot.ylabel(yaxlabel1, fontsize=24)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    pyplot.title(title1, fontsize=16)
    pyplot.text(-0.05, 1.05, 'A', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ax = panel(0, 1)
    sleep_panel(traitSummary2, trait=tr, sub=True, title=False)
    pyplot.ylabel(yaxlabel2, fontsize=24)
    pyplot.xticks(fontsize=18)
    pyplot.yticks([0.2, 0.4, 0.6, 0.8], fontsize=18)
    pyplot.text(-0.05, 1.05, 'B', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    pyplot.title(title2, fontsize=16)

    pyplot.xlabel("Generation", fontsize=24)

    fig1AB.savefig(savepath) if (savepath != "") else None

    return None


def traits(traitSummary1, traitSummary2, traitlist, yaxlabels=[], panelabels=[], savepath=""):
    """ plot composite figure for multiple traits with two summaries in side-by-side panels """

    if type(traitlist) == str:
        newlist = [traitlist]
    elif type(traitlist) == list:
        newlist = traitlist
    else:
        raise( ValueError("argument is not a string or list of strings") )

    L = len(newlist)
    panel = lambda panx, pany: pyplot.subplot2grid( (L,2), (panx, pany) )

    figS2 = figure( figsize = (9, 3*L) )

    for i,tr in enumerate(newlist):
        ax = panel(i, 0)
        sleep_panel(traitSummary1, trait=tr, sub=True, title=False)
        pyplot.ylabel(yaxlabels[i], fontsize=10) if len(yaxlabels)>0 else pyplot.ylabel(tr, fontsize=12)
        # pyplot.title("$Mean$", fontsize=14) if (i==0) else None
        pyplot.text(-0.15, 1.02, panelabels[i][0], fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes) if len(panelabels)>0 else None

        ax = panel(i, 1)
        sleep_panel(traitSummary2, trait=tr, sub=True, title=False)
        pyplot.text(-0.12, 1.02, panelabels[i][1], fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes) if len(panelabels)>0 else None
        pyplot.ylabel(yaxlabels[i]+" $CV_E$", fontsize=10) if len(yaxlabels)>0 else pyplot.ylabel(tr, fontsize=12)

    pyplot.xlabel("Generation", fontsize=14)

    figS2.savefig(savepath) if (savepath != "") else None

    return None


def heritability_scatter( ΣS, ΣR, colour ):
    """ simple scatter plot for heritability data """
    plot(ΣS, ΣR, 'o', color=colour)

    return None


def heritability_regress( ΣS, ΣR, colour ):
    """ compute regression for heritability selection differentials and plot regresion line """

    slope, intercept, r2, pvalue, std = scipy.stats.linregress(ΣS, ΣR)
    xspan = max(ΣS) - min(ΣS)
    x1 = min(ΣS) - 0.2*xspan
    x2 = max(ΣS) + 0.2*xspan

    y1 = intercept + x1*slope
    y2 = intercept + x2*slope

    plot( [x1,x2], [y1,y2], color=colour )

    return slope, intercept, r2, pvalue, std


def heritability( ΣS, ΣR, groups=[["L1", "L2"],["S1", "S2"], ["C1", "C2"]], colours=[["LightGreen", "ForestGreen"], ["DarkOrange", "Red"], ['0.7', '0']], titles=['long','short','control'], savepath="", verbo=True):

    """ plot composite heritability figure with data and regerssion lines for all selection schemes """

    regressions = pandas.DataFrame(index=["slope", "intercept", "r2", "pvalue", "std"], columns=ΣS.columns)

    L = len(groups)
    panel = lambda panx, pany: pyplot.subplot2grid( (1,3), (panx, pany) )

    panelabels = ['C', 'D', 'E']
    fig1CDE = figure( figsize=(18,6) )
    for j,grp in enumerate(groups):

        ax = panel(0,j)
        for i,linn in enumerate(grp):

            heritability_scatter( ΣS[linn], ΣR[linn], colour=colours[j][i] )

            slope, intercept, r2, pvalue, std = heritability_regress( ΣS[linn], ΣR[linn], colour=colours[j][i] )
            regressions.loc[["slope", "intercept", "r2", "pvalue", "std"], linn] = slope, intercept, r2, pvalue, std

            if (verbo==True):
                print(">>> Line: ", titles[j], i)
                print(">>> h^2 (slope):", slope)
                print(">>> R^2:", r2)
                print(">>> std:", std)
                print(">>> p-value:", pvalue)
                print("")

        pyplot.xlabel("$\Sigma S$", fontsize=20) if (j<2) else pyplot.xlabel("$\Sigma D$", fontsize=18) if (j==2) else None
        pyplot.ylabel("$\Sigma R$", fontsize=20) if (j==0) else None

        pyplot.title(titles[j], fontsize=22) if (len(titles)>0) else None
        pyplot.text(-0.05, 1.05, panelabels[j], fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)#  if len(panelabels)>0 else None

    if (savepath != ""):
        print(">>> saving files")
        fig1CDE.savefig(savepath+".png")
        regressions.to_csv(savepath+".csv", sep=",", header=True)
    else:
        print(">>> no files saved")

    return regressions


def heritability_stack( ΣS, ΣR, groups=[["L1", "L2"],["S1", "S2"], ["C1", "C2"]], colours=[["LightGreen", "ForestGreen"], ["DarkOrange", "Red"], ['0.7', '0']], titles=['long','short','control'], savepath="", verbo=False):

    """ same as `heritability` function, but instead of side-by-side it stacks panels into 2x2 grid """


    L = len(groups)
    panel = lambda panx, pany: pyplot.subplot2grid( (2,2), (panx, pany) )

    panelabels = ['C', 'D', 'E']
    fig1CDE = figure( figsize=(12,12) )
    for j,grp in enumerate(groups):

        ax = panel(0,j) if (j<2) else panel(1,0)
        for i,linn in enumerate(grp):

            heritability_scatter( ΣS[linn], ΣR[linn], colour=colours[j][i] )

            slope, intercept, r2, pvalue, std = heritability_regress( ΣS[linn], ΣR[linn], colour=colours[j][i] )

        pyplot.xlabel("$\Sigma D$", fontsize=18) if (j==2) else None
        pyplot.xlabel("$\Sigma S$", fontsize=18) if (j==1) else None
        pyplot.ylabel("$\Sigma R$", fontsize=18) if (j!=1) else None

        pyplot.title(titles[j], fontsize=20) if (len(titles)>0) else None
        pyplot.text(-0.05, 1.05, panelabels[j], fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)#  if len(panelabels)>0 else None

    fig1CDE.savefig(savepath) if (savepath != "") else None

    return None
