#!/usr/bin/python3

"""
Phenotypic analysis of artifical selection experiment
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

#import evorna.phenotypic.summaries as phenosummaries
#import evorna.phenotypic.phenotype_plots as phenoplots

from evorna.phenotypic.summaries import meta_data, trait_mean_frame, trait_variance_frame, trait_cv_frame, breeders_sigma
from evorna.phenotypic.phenotype_plots import traits as plot_traits, one_trait as plot_one_trait, heritability as plot_heritability, show

# set project directories
project_dir = os.getcwd() # assumes this script is being run from one level above the `evorna` package
data_dir = os.path.join( project_dir, "data" ) # specifies data_dir inside of project_dir
results_dir = os.path.join( os.path.expanduser("~"), "results_local", "evorna", "glm")

# load data and metadata from DAM files
metaDataExperimental, experimentalPhenotypicData = meta_data(data_dir, "phenotypic_data.csv", samples=False)
metaDataRNASamples, rnasamplePhenotypicData = meta_data(data_dir, "phenotypic_data_for_RNA.csv", samples=True)

# compute traits mean and CV_E
traitMeans = trait_mean_frame(experimentalPhenotypicData, metaDataExperimental, experimentalPhenotypicData.columns)
traitCVs = trait_cv_frame(experimentalPhenotypicData, metaDataExperimental, experimentalPhenotypicData.columns)

# trait variances can also be computed
#traitVariances = trait_variance_frame(experimentalPhenotypicData, metaDataExperimental, experimentalPhenotypicData.columns)

# Specify y axes  titles and labels of figures
yaxlabels = ["Day bout number", "Day sleep (min.)", "Night bout number", "Night Sleep (min.)", "Waking activity (cts./min.)", "Sleep latency (min.)", "Day avg. bout length (min.)", "Night avg. bout length (min.)"]

panelabels = [['A', 'B'], ['C', 'D'], ['E', 'F'], ['G', 'H']]
panelabels2 = [['I', 'J'], ['K', 'L'], ['M', 'N'], ['O', 'P']]

# plot "Figure S2" panels as two separate figures
plot_traits( traitMeans, traitCVs, list(experimentalPhenotypicData.columns)[0:4], yaxlabels=yaxlabels[0:4], panelabels=panelabels, savepath=os.path.join(project_dir, "figures", "figS2.png") )

plot_traits( traitMeans, traitCVs, list(experimentalPhenotypicData.columns)[4:8], yaxlabels=yaxlabels[4:8], panelabels=panelabels2, savepath=os.path.join(project_dir, "figures", "figS2 (continued).png") )


# plot top part of "Figure 1" (panels A and B): Night Sleep summaries
plot_one_trait(traitMeans[['Sel', 'Rep', 'Generation', 'sleepn']] , traitCVs[['Sel', 'Rep', 'Generation', 'sleepn']] , "sleepn", yaxlabels=["Night Sleep Mean (minutes)", "Night Sleep $CV_E$"], savepath=os.path.join(project_dir, "figures", "fig1AB.png") )


# compute heritability
ΣS, ΣR, ΔS, ΔR, nextgenmeans, linemeans = breeders_sigma(metaDataExperimental, experimentalPhenotypicData, trait="sleepn", line="Line", generation="Generation", intermediates=True)


# plot bottom part of "Figure 1" (panels C,D, and E): heritability plots
regressions = plot_heritability( ΣS, ΣR, groups=[["L1", "L2"],["S1", "S2"], ["C1", "C2"]], colours=[["LightGreen", "ForestGreen"], ["DarkOrange", "Red"],['0.7', '0']], titles=['long', 'short','control'], savepath=os.path.join(project_dir, "figures", "fig1CDE")  )

# show plots
show()
