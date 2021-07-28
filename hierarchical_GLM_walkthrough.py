#/usr/bin/python3

import os
import numpy

#import pystan
from cmdstanpy import cmdstan_path

import pandas
import pickle
# from numpy.random import randint, poisson as rpoisson, normal as rnormal
import scipy.stats
import statsmodels.sandbox.stats.multicomp


import timeit

import evorna.rnaseq as rnaseq
from evorna.diffx import design, model
from evorna.diffx.hglm_plot import hglm_plot, show
from evorna.diffx.bayesian import hierarchical_glm_pystan as glm

#import evorna.diffx as hglm

project_dir = os.getcwd() # assumes this script is being run from one level above the `evorna` package
data_dir = os.path.join( project_dir, "data" ) # specifies data_dir inside of project_dir
results_dir = os.path.join( project_dir, "results" )

os.chdir(project_dir) # will allow loading

genic_file = os.path.join( data_dir, "genic.csv" )
intergenic_file = os.path.join( data_dir, "intergenic.csv" )
metadata_file = os.path.join( data_dir, "metadata.csv" )
flybase_file = os.path.join( data_dir, "Flybase_ID_gene_info_111717.csv" )


# loading and checking metadata and filtered data for specific sex
print(">>> loading data...")
metaDataFemaleUnchecked, data = rnaseq.filtered_sex_data(data_dir, metadata_file, genic_file, intergenic_file, "Female", normalized=False)
metadata = rnaseq.check_axis_consistency(metaDataFemaleUnchecked)

flyBaseInfo = pandas.read_csv( flybase_file, sep=",", header=0, index_col=0 )

print(">>> Design matrix and label check:")
# creates design frame for full and reduced model using metadata
designFrame, reducedDesign = design.create_design_matrix(metadata)

# check labels of both design matrices against data
designFrame, reducedDesign = design.check_design_labels(data, designFrame, reducedDesign)

# read and compile Stan model in cmdstan path
model_file_path = os.path.join( project_dir, "stan", "glm_nbinom_hierarchical.stan")
reduced_model_file_path = os.path.join( project_dir, "stan", "glm_nbinom_hiereduced.stan" )

# create data frame to hold GLM results
symbolFrame, resultsFrame = design.create_results_frames( data[ list(data.iloc[:,0:7].columns) + ["FBgn0031141"] ] , flyBaseInfo, verbo=True )
# symbolFrame, resultsFrame = design.create_results_frames( data["FBgn0031141"] , flyBaseInfo, verbo=True )

# individual genes can be fit with the GLM model by providing them with the FlyBase ID which will identify the column with its expresion data...
resj = glm.fit( fbid="FBgn0031141", data=data, metadata=metadata, designFrame=designFrame, reducedDesign=reducedDesign, models=[model_file_path, reduced_model_file_path] )  #, outpath=os.path.join(results_dir, "traces") )

# results for one gene are one row of a table (i.e. supplementary tables S5 and S6), here we assign results computed above to that row.
# This can be done in a single step by assigning the `glm.fit()` output to `resultsFrame.loc[fbid]``
resultsFrame.loc["FBgn0031141"] = resj.values


parameterFrame, fullPosteriorMeanFrame, fullPosterior25Frame, fullPosterior975Frame, reducedPosteriorMeanFrame, reducedPosterior25Frame, reducedPosterior975Frame = design.split_results_frame(metadata, resultsFrame)  #, results=results_dir)

hglm_plot("FBgn0031141", "Female", metadata, data, parameterFrame, fullPosteriorMeanFrame, fullPosterior25Frame, fullPosterior975Frame, reducedPosteriorMeanFrame, reducedPosterior25Frame, reducedPosterior975Frame, symbolFrame=symbolFrame, colors=["DarkOrange", "MediumAquamarine"], savepath=os.path.join(project_dir, "figures", "GLM_Female_FBgn0031141.png")  )
show()


# helper lambda-function fixes data, metadata and design arguments as data frames loaded above, and requires only only fbid argument to determine which gene is fitted with GLM model
fit_one_gene = lambda fbid: glm.fit( fbid, data, metadata, designFrame, reducedDesign, models=[model_file_path, reduced_model_file_path] )  # , outpath=os.path.join( results_dir, "traces") )

fbidList = list( resultsFrame.index )

# and loop over several genes
tic = timeit.default_timer()
print(">>> Fitting GLMs:")
for fbid in fbidList:
    resultsFrame.loc[fbid, "bIntercept1":] = fit_one_gene(fbid)
tac = timeit.default_timer()
print(">>> looping series of genes run time:", tac-tic, "seconds \n")

# Save table
resultsFrame.to_csv( os.path.join( project_dir, "tables", "Table S5 - HGLM - Females.csv" ), header=True, index=True, sep="," )
