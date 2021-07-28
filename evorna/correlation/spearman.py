#!/usr/bin/python3

"""
Computes correlations between RNA expression of pairs of genes
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import numpy
import pandas

import scipy.stats
from scipy.stats import spearmanr, pearsonr

from statsmodels.sandbox.stats.multicomp import multipletests


def fisher_transformation(r):
    """ Fisher Transformation function for correlation coefficient `r`, which yields a distribution that is approximately normal with variance σ^2 ~ 1/(N-3) [ see for instance Ruscio (2008) ]"""

    F = (1/2) * numpy.log( (1+r)/(1-r) )

    return F


def confidence_limits(ρ, N, α=0.05):
    """ function to compute transformation of correlation coefficient and its confidence limits, and inverse transformation """

    # Fisher transformation of correlation coefficient estimate:
    z_ρ = fisher_transformation(ρ)

    # Confidence Limites for the approximation of normal distibution with variance described above
    z_u = z_ρ + numpy.sqrt( 1/(N-3) ) * scipy.stats.norm(0,1).ppf(1-α/2)
    z_l = z_ρ + numpy.sqrt( 1/(N-3) ) * scipy.stats.norm(0,1).ppf(α/2)

    # inverse transformation back into correlation coefficient
    r_u = (numpy.exp(2*z_u) - 1)/(numpy.exp(2*z_u) + 1)
    r_l = (numpy.exp(2*z_l) - 1)/(numpy.exp(2*z_l) + 1)

    return [r_l, r_u]


def confidence_limits_vector(rvec, N, α=0.05):
    """ vectorized version of function above to list of correlation coefficients """

    spearmap = lambda r : confidence_limits(r, N, α)

    rlims = list( map(spearmap, rvec) )

    return rlims


def correlation_dataframe(symbolFrame, subdataFemale, subdataMale, metaDataFem, metaDataMale):
    """ looping function to compute spearman correlation coefficients over all pairs of expression vectors """

    corrFrame = pandas.DataFrame( index = [ fbidi+fbidj for i,fbidi in enumerate(symbolFrame.index) for j,fbidj in enumerate(symbolFrame.index) if i<j ], columns = ['fbid_one', 'fbid_two', 'gene_one', 'gene_two', 'corrFemShort', 'corrFemControl', 'corrFemLong', 'corrMaleShort', 'corrMaleControl', 'corrMaleLong', 'pFemShort', 'pFemControl', 'pFemLong', 'pMaleShort', 'pMaleControl', 'pMaleLong', 'bhFemShort', 'bhFemControl', 'bhFemLong', 'bhMaleShort', 'bhMaleControl', 'bhMaleLong'] )

    corrFrame[['fbid_one', 'fbid_two']] = [[fbidi,fbidj] for i,fbidi in enumerate(symbolFrame.index) for j,fbidj in enumerate(symbolFrame.index) if i<j ]

    corrFrame[['gene_one', 'gene_two']] = [ [symbolFrame.loc[fbidi], symbolFrame.loc[fbidj] ] for i,fbidi in enumerate(symbolFrame.index) for j,fbidj in enumerate(symbolFrame.index) if i<j ]

    for i,idx in enumerate(corrFrame.index):
        fbidi = corrFrame.loc[idx, 'fbid_one']
        fbidj = corrFrame.loc[idx, 'fbid_two']

        print(">>>", i, ": ", fbidi, fbidj) if ((i % 100) == 0) else None

        corrFrame.loc[idx, ['corrFemShort', 'pFemShort'] ] = spearmanr( subdataFemale.loc[metaDataFem.sel=="short", fbidi ].values, subdataFemale.loc[metaDataFem.sel=="short", fbidj ].values )

        corrFrame.loc[idx, ['corrFemControl', 'pFemControl'] ] = spearmanr( subdataFemale.loc[metaDataFem.sel=="control", fbidi ].values, subdataFemale.loc[metaDataFem.sel=="control", fbidj ].values )

        corrFrame.loc[idx, ['corrFemLong', 'pFemLong'] ] = spearmanr( subdataFemale.loc[metaDataFem.sel=="long", fbidi ].values, subdataFemale.loc[ metaDataFem.sel=="long", fbidj ].values )

        corrFrame.loc[idx, ['corrMaleShort', 'pMaleShort'] ] = spearmanr( subdataMale.loc[metaDataMale.sel=="short", fbidi ].values, subdataMale.loc[metaDataMale.sel=="short", fbidj ].values )

        corrFrame.loc[idx, ['corrMaleControl', 'pMaleControl'] ] = spearmanr( subdataMale.loc[metaDataMale.sel=="control", fbidi ].values, subdataMale.loc[metaDataMale.sel=="control", fbidj ].values )

        corrFrame.loc[idx, ['corrMaleLong', 'pMaleLong'] ] = spearmanr( subdataMale.loc[metaDataMale.sel=="long", fbidi ].values, subdataMale.loc[ metaDataMale.sel=="long", fbidj ].values )

        corrFrame['bhFemShort' ] = multipletests(corrFrame['pFemShort' ].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

        corrFrame['bhFemControl' ] = multipletests(corrFrame['pFemControl' ].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

        corrFrame['bhFemLong' ] = multipletests(corrFrame['pFemLong' ].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

        corrFrame['bhMaleShort' ] = multipletests(corrFrame['pMaleShort' ].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

        corrFrame['bhMaleControl' ] = multipletests(corrFrame['pMaleControl' ].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

        corrFrame['bhMaleLong' ] = multipletests(corrFrame['pMaleLong' ].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

    return corrFrame


def confidence_limits_dataframe(spearFrame, N, alpha=0.05):

    spearFrameCI = pandas.concat( [ spearFrame, pandas.DataFrame( index=spearFrame.index, columns=['lowerFemShort', 'upperFemShort', 'lowerFemControl', 'upperFemControl', 'lowerFemLong', 'upperFemLong', 'lowerMaleShort', 'upperMaleShort', 'lowerMaleControl', 'upperMaleControl', 'lowerMaleLong', 'upperMaleLong'] ) ], axis=1 )

    spearFrameCI[ ['lowerFemShort', 'upperFemShort'] ]  = confidence_limits_vector(spearFrame['corrFemShort'], N, α=alpha)
    spearFrameCI[ ['lowerFemControl', 'upperFemControl'] ] = confidence_limits_vector(spearFrame['corrFemControl'], N, α=alpha)
    spearFrameCI[ ['lowerFemLong', 'upperFemLong'] ] = confidence_limits_vector(spearFrame['corrFemLong'], N, α=alpha)

    spearFrameCI[ ['lowerMaleShort', 'upperMaleShort'] ] = confidence_limits_vector(spearFrame['corrMaleShort'], N, α=alpha)
    spearFrameCI[ ['lowerMaleControl', 'upperMaleControl'] ] = confidence_limits_vector(spearFrame['corrMaleControl'], N, α=alpha)
    spearFrameCI[ ['lowerMaleLong', 'upperMaleLong'] ] = confidence_limits_vector(spearFrame['corrMaleLong'], N, α=alpha)

    return spearFrameCI
