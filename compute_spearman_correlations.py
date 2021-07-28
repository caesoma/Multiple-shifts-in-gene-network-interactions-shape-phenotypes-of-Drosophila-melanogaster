#!/usr/bin/python3

"""
Computes correlations between RNA expression of pairs of genes
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

import numpy
import pandas

import warnings
import timeit

import os
import numpy

import evorna.rnaseq as rnaseq
import evorna.spearman as spearmanf


project_dir = os.getcwd() # assumes this script is being run from one level above the `evorna` package
data_dir = os.path.join( project_dir, "data" ) # assumes data and results directories are inside `project_dir`, but they can be loaded from anywhere
results_dir = os.path.join( project_dir, "results" )

genic_file = os.path.join( data_dir, "genic.csv" )
intergenic_file = os.path.join( data_dir, "intergenic.csv" )
metadata_file = os.path.join( data_dir, "metadata.csv" )
flybase_file = os.path.join( data_dir, "Flybase_ID_gene_info_111717.csv" )

# load data
metaDataFem, dataNormFemale = rnaseq.filtered_sex_data(data_dir, metadata_file, genic_file, intergenic_file, "Female", normalized=True)
metaDataMale, dataNormMale = rnaseq.filtered_sex_data(data_dir, metadata_file, genic_file, intergenic_file, "Male", normalized=True)


# load GLM results table
resFemale = pandas.read_csv( os.path.join( results_dir, "glm", "hierarchic", "results_parallel_hierarchicstan_allfemale.csv"), sep=",", header=0, index_col=0)
resMale = pandas.read_csv( os.path.join( results_dir, "glm", "hierarchic", "results_parallel_hierarchicstan_allmale.csv" ), sep=",", header=0, index_col=0)


# restrict genes to those under a specific level of significance (1e-3)
sublist = list( resFemale[ (resFemale.padj<1e-3) & (resMale.padj<1e-3) ].sort_values('padj', axis=0, ascending=True).index )

subdataNormFemale = dataNormFemale[sublist] # substting the data to those genes only
subdataNormMale= dataNormMale[sublist]

symbolist = resFemale.loc[sublist].symbol

# compute spearman correlations
spearFrame = spearmanf.correlation_dataframe( symbolist, subdataNormFemale, subdataNormMale, metaDataFem, metaDataMale )

spearFrameCI = spearmanf.confidence_limits_dataframe( spearFrame, N=subdataNormFemale.loc[ metaDataFem.sel=="short", subdataNormFemale.columns[0] ].shape[0], alpha=0.05 )
