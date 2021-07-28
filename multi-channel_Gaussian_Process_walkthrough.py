#/usr/bin/python3

import os
import numpy

#import pystan
from cmdstanpy import cmdstan_path, CmdStanModel

import pandas
import pickle
# from numpy.random import randint, poisson as rpoisson, normal as rnormal
import scipy.stats
import statsmodels.sandbox.stats.multicomp

from matplotlib.pyplot import show

import timeit

import evorna.rnaseq as rnaseq
from evorna.nonlinear import cmdstanify, bayesian
from evorna.nonlinear.gaussian_processes import single, multi, gp_plot

project_dir = os.getcwd() # assumes this script is being run from one level above the `evorna` package
data_dir = os.path.join( project_dir, "data" ) # specifies data_dir inside of project_dir (can be anywhere else)
results_dir = os.path.join( project_dir, "results" )

os.chdir(project_dir) # change working directory to `project_dir` if that is different from the current working directory

genic_file = os.path.join( data_dir, "genic.csv" )
intergenic_file = os.path.join( data_dir, "intergenic.csv" )
metadata_file = os.path.join( data_dir, "metadata.csv" )
flybase_file = os.path.join( data_dir, "Flybase_ID_gene_info_111717.csv" )


# loading and checking metadata and filtered data for specific sex
print(">>> loading data...")
metaDataFemaleUnchecked, data = rnaseq.filtered_sex_data(data_dir, metadata_file, genic_file, intergenic_file, "Female", normalized=False)
metadata = rnaseq.check_axis_consistency(metaDataFemaleUnchecked)

flybase_file = os.path.join( data_dir, "Flybase_ID_gene_info_111717.csv" )
flyBaseInfo = pandas.read_csv( flybase_file, sep=",", header=0, index_col=0 )

subdata = data[ ["FBgn0004426", "FBgn0031141"] ]

# read and compile Stan model in cmdstan folder (unlike the GLM analysis, this requires cmdStan, some straightforward modifications can be made to the functions to run with PyStan, but without them they won't work )
single_gp_path = os.path.join( cmdstan_path(), "models", "gaussianprocesses", "gaussianprocess_lognbinom_single.stan" )
single_gp_model = CmdStanModel( stan_file=single_gp_path )

# create CmdStan dictionaries with data and algorithm parameters for all selection schemes
cmdstanDict = cmdstanify.write_several_singles( modelName="gaussianprocess_lognbinom_single", subdata=subdata, metadata=metadata, N=len(set(metadata.generation)), chains=8, iterations=-1, burn=-1, savewarmup=0, samples=1000, folder=results_dir, verbo=False)

# read Stan data diciontary (the `write_several_singles` function writes to disk, but does not return all dicionaries created)

# "gene 1" (here FBgn0031141)
shortFemaleDict1 = cmdstanify.read_json_file( os.path.join(results_dir, "short_Female_FBgn0031141_data.json") )  # short selection scheme
controlFemaleDict1 = cmdstanify.read_json_file( os.path.join(results_dir, "control_Female_FBgn0031141_data.json") )
longFemaleDict1 = cmdstanify.read_json_file( os.path.join(results_dir, "long_Female_FBgn0031141_data.json") )

# "gene 2"
shortFemaleDict2 = cmdstanify.read_json_file( os.path.join(results_dir, "short_Female_FBgn0004426_data.json") )
controlFemaleDict2 = cmdstanify.read_json_file( os.path.join(results_dir, "control_Female_FBgn0004426_data.json") )
longFemaleDict2 = cmdstanify.read_json_file( os.path.join(results_dir, "long_Female_FBgn0004426_data.json") )


# run Bayesian inference for each selection scheme for both genes:
# FBgn0031141
print(">>> beginning single-channel sampling:")
tic = timeit.default_timer()
singleShortGPsamples1 = single_gp_model.sample(data=shortFemaleDict1, iter_sampling=cmdstanDict["iterations"], iter_warmup=cmdstanDict["burnin"], save_warmup=True, inits={'KfDiag': 1}, thin=cmdstanDict["thinning"], chains=cmdstanDict["alice"], parallel_chains=cmdstanDict["alice"], output_dir= os.path.join(results_dir, "traces") )

singleControlGPsamples1 = single_gp_model.sample(data=controlFemaleDict1, iter_sampling=cmdstanDict["iterations"], iter_warmup=cmdstanDict["burnin"], save_warmup=True, inits={'KfDiag': 1}, thin=cmdstanDict["thinning"], chains=cmdstanDict["alice"], parallel_chains=cmdstanDict["alice"], output_dir=os.path.join(results_dir, "traces") )

singleLongGPsamples1 = single_gp_model.sample(data=longFemaleDict1, iter_sampling=cmdstanDict["iterations"], iter_warmup=cmdstanDict["burnin"], save_warmup=True, inits={'KfDiag': 1}, thin=cmdstanDict["thinning"], chains=cmdstanDict["alice"], parallel_chains=cmdstanDict["alice"], output_dir=os.path.join(results_dir, "traces") )

# FBgn0004426
singleShortGPsamples2 = single_gp_model.sample(data=shortFemaleDict2, iter_sampling=cmdstanDict["iterations"], iter_warmup=cmdstanDict["burnin"], save_warmup=True, thin=cmdstanDict["thinning"], chains=cmdstanDict["alice"], parallel_chains=cmdstanDict["alice"], output_dir=os.path.join(results_dir, "traces") )

singleControlGPsamples2 = single_gp_model.sample(data=controlFemaleDict2, iter_sampling=cmdstanDict["iterations"], iter_warmup=cmdstanDict["burnin"], save_warmup=True, inits={'KfDiag': 1}, thin=cmdstanDict["thinning"], chains=cmdstanDict["alice"], parallel_chains=cmdstanDict["alice"], output_dir=os.path.join(results_dir, "traces") )

singleLongGPsamples2 = single_gp_model.sample(data=longFemaleDict2, iter_sampling=cmdstanDict["iterations"], iter_warmup=cmdstanDict["burnin"], save_warmup=True, inits={'KfDiag': 1}, thin=cmdstanDict["thinning"], chains=cmdstanDict["alice"], parallel_chains=cmdstanDict["alice"], output_dir=os.path.join(results_dir, "traces") )

tac = timeit.default_timer()
print(">>> computation time:", (tac-tic)/60, "minutes")

# extract posterior distribution/traces from sampling object
# gene 1
singlesShortPosterior1, _ = bayesian.posterior_from_samples(singleShortGPsamples1, cmdstanDict)  # singleShortGPsamples1.stan_variables()
singlesControlPosterior1, _ = bayesian.posterior_from_samples(singleControlGPsamples1, cmdstanDict)
singlesLongPosterior1, _ = bayesian.posterior_from_samples(singleLongGPsamples1, cmdstanDict)

# gene 2
singlesShortPosterior2, _ = bayesian.posterior_from_samples(singleShortGPsamples2, cmdstanDict)
singlesControlPosterior2, _ = bayesian.posterior_from_samples(singleControlGPsamples2, cmdstanDict)
singlesLongPosterior2, _ = bayesian.posterior_from_samples(singleLongGPsamples2, cmdstanDict)


gPosterior1 = single.evorna_single_gp_posterior(results_dir, cmdstanDict, "FBgn0031141", subdata["FBgn0031141"], metadata, posterior=[singlesShortPosterior1, singlesControlPosterior1, singlesLongPosterior1], verbo=True)

gp_plot.gp_manuscript_single_plot(save_path=os.path.join( project_dir, "figures", "gp_"+"FBgn0031141"+".png" ), symbol="CG1304", fbidata=subdata["FBgn0031141"], metadata=metadata, gPosterior=gPosterior1, color="RoyalBlue")


# summarize means and variances of single channel parameters (to do this over many pairs a new function with a simple loop would be needed to do this for all of them)
singleChannelSummaries = ["shortKfDiagMean", "shortKfDiagSigma", "shortEllMean", "shortEllSigma", "shortAlphaMean", "shortAlphaSigma", "controlKfDiagMean", "controlKfDiagSigma", "controlEllMean", "controlEllSigma", "controlAlphaMean", "controlAlphaSigma", "longKfDiagMean", "longKfDiagSigma", "longEllMean", "longEllSigma", "longAlphaMean", "longAlphaSigma"]

pairiors = pandas.DataFrame( index=["FBgn0031141", "FBgn0004426"], columns=singleChannelSummaries )

pairiors.loc["FBgn0031141"] = bayesian.summarize_gp_dictionary( "FBgn0031141", [singlesShortPosterior1, singlesControlPosterior1, singlesLongPosterior1] ).values
pairiors.loc["FBgn0004426"] = bayesian.summarize_gp_dictionary( "FBgn0004426", [singlesShortPosterior2, singlesControlPosterior2, singlesLongPosterior2] ).values



# Dual(multi)-channel model analysis

# read and compile Stan multi-channel model in cmdstan path
multi_gp_path = os.path.join( cmdstan_path(), "models", "gaussianprocesses", "gaussianprocess_lognbinom2.stan")
multi_gp_model = CmdStanModel( stan_file=multi_gp_path )


# Write json files with data from pair of genes
cmdstanMultiDict = cmdstanify.write_several_pairs(results_dir, modelName="gaussianprocess_lognbinom2", subdata=subdata, metadata=metadata, N=len(set(metadata.generation)), hyperDataFrame=pairiors, chains=8, iterations=-1, burn=-1, savewarmup=0, samples=1000, data_dir=results_dir, verbo=False)

# read dictionaries from disk for each selection scheme
shortFemaleMulti = cmdstanify.read_json_file( os.path.join(results_dir, "short_Female_FBgn0004426_FBgn0031141_data.json") )
controlFemaleMulti = cmdstanify.read_json_file( os.path.join(results_dir, "control_Female_FBgn0004426_FBgn0031141_data.json") )
longFemaleMulti = cmdstanify.read_json_file( os.path.join(results_dir, "long_Female_FBgn0004426_FBgn0031141_data.json") )

# write json file with initial values for Kf, to ensure the chain starts from a positive definite value
cmdstanify.write_json_file( os.path.join(results_dir, "gp2_init.json"), {"KfTril": [0.001], "KfDiag": [1.0, 1.0]} ) # once written, this step doesn't need to be done in every analysis, just read the same file, as in the next line
init_dict = cmdstanify.read_json_file( os.path.join(results_dir, "gp2_init.json") )


print(">>> beginning dual-channel sampling:")
tic = timeit.default_timer()
multiShortGPsamples = multi_gp_model.sample(data=shortFemaleMulti, iter_sampling=cmdstanMultiDict["iterations"], iter_warmup=cmdstanMultiDict["burnin"], save_warmup=cmdstanMultiDict['washburn'], inits=init_dict, thin=cmdstanMultiDict["thinning"], chains=cmdstanMultiDict["alice"], parallel_chains=cmdstanMultiDict["alice"], output_dir=os.path.join(results_dir, "traces"))

multiControlGPsamples = multi_gp_model.sample(data=controlFemaleMulti, iter_sampling=cmdstanMultiDict["iterations"], iter_warmup=cmdstanMultiDict["burnin"], save_warmup=cmdstanMultiDict['washburn'], inits=init_dict, thin=cmdstanMultiDict["thinning"], chains=cmdstanMultiDict["alice"], parallel_chains=cmdstanMultiDict["alice"], output_dir=os.path.join(results_dir, "traces"))

multiLongGPsamples = multi_gp_model.sample(data=longFemaleMulti, iter_sampling=cmdstanMultiDict["iterations"], iter_warmup=cmdstanMultiDict["burnin"], save_warmup=cmdstanMultiDict['washburn'], inits=init_dict, thin=cmdstanMultiDict["thinning"], chains=cmdstanMultiDict["alice"], parallel_chains=cmdstanMultiDict["alice"], output_dir=os.path.join(results_dir, "traces"))
tac = timeit.default_timer()
print(">>> computation time:", (tac-tic)/60, "minutes")

# get posteriors from sampling object
multiShortGPposterior, _ = bayesian.posterior_from_samples(multiShortGPsamples, cmdstanMultiDict)
multiControlGPposterior, _ = bayesian.posterior_from_samples(multiControlGPsamples, cmdstanMultiDict)
multiLongGPposterior, _ = bayesian.posterior_from_samples(multiLongGPsamples, cmdstanMultiDict)

# compute posterior of Gaussian Process output (from samples/posterior of individual model parameters)
multigPosterior, rho, stars = multi.evorna_gp_posterior(results_dir, cmdstanMultiDict, fbidij=["FBgn0004426", "FBgn0031141"], subdata=subdata, metadata=metadata, posterior=[multiShortGPposterior, multiControlGPposterior, multiLongGPposterior], verbo=False)

gp_plot.gp_manuscript_pair_plot(save_path=os.path.join( project_dir, "figures", "gp_"+"FBgn0004426"+"_"+"FBgn0031141"+".png" ), symbolist=flyBaseInfo.loc[["FBgn0004426", "FBgn0031141"]].SYMBOL, fbidata2=subdata, metadata=metadata, gPosterior=multigPosterior, rho=rho, significance=stars, color=["RoyalBlue", "Crimson"])


# heatmap of correlations can be assembled from each pairwise estimate; the `compute_Cf_ij` function will calculate the correlation from the signal covariances (parameter multiplying any off-diagonal block) -- in the case of 2 genes there is only one off-diagonal term; for 85 genes there are 3570 combinations, each corresponding to a run like the above
cfMeanij, cf25ij, cf975ij = bayesian.compute_Cf_ij(dictraces=(multiShortGPposterior,multiControlGPposterior,multiLongGPposterior), fbidij=["FBgn0004426", "FBgn0031141"], burn=0, verbo=False)

# a matrix can be assembled from each off-diagonal term, here a 2x2 matrix with 1 on all diagonal terms
corrMatrixShort, corrMatrixControl, corrMatrixLong = numpy.eye(2), numpy.eye(2), numpy.eye(2)
corrMatrixShort[0,1], corrMatrixControl[0,1], corrMatrixLong[0,1]  = corrMatrixShort[1,0], corrMatrixControl[1,0], corrMatrixLong[1,0] = cfMeanij["short"], cfMeanij["control"], cfMeanij["long"]

# The `plotK_all2` function requires data frames (because of its index/column labels) and will plot a heatmap, here is a minimal version of mansucript figure 4
gp_plot.plotK({"short": pandas.DataFrame(corrMatrixShort, index=["FBgn0004426", "FBgn0031141"], columns=["FBgn0004426", "FBgn0031141"]), "control":pandas.DataFrame(corrMatrixControl, index=["FBgn0004426", "FBgn0031141"], columns=["FBgn0004426", "FBgn0031141"]), "long":pandas.DataFrame(corrMatrixLong, index=["FBgn0004426", "FBgn0031141"], columns=["FBgn0004426", "FBgn0031141"])}, filepath=os.path.join( project_dir, "figures", "Kmatrix_"+"FBgn0004426"+"_"+"FBgn0031141"+".png" ), correlation=True);

show()
