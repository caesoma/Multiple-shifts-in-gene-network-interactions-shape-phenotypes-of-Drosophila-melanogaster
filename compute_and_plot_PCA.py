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

import os

import numpy
import pandas

import os
import numpy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from matplotlib.pyplot import figure, plot, show
import matplotlib.pyplot as pyplot

import evorna.rnaseq as rnaseq

# set project and data directories
project_dir = os.getcwd()
data_dir = os.path.join( project_dir, "data" )

genic_file = os.path.join( data_dir, "genic.csv" )
intergenic_file = os.path.join( data_dir, "intergenic.csv" )
metadata_file = os.path.join( data_dir, "metadata.csv" )
flybase_file = os.path.join( data_dir, "Flybase_ID_gene_info_111717.csv" )

# load normalized data
metadata, normData, _ = rnaseq.normalized_data(data_dir, metadata_file, genic_file, intergenic_file)

# scikit learn PCA
dataNormStd = StandardScaler().fit_transform( normData )

N = 10
pca = PCA(n_components=N)
pcn = pca.fit_transform( dataNormStd )

pcDataFrame = pandas.DataFrame( data = pcn, index=normData.index, columns = [ 'pc' + str(n+1) for n in range(N)] )
pcentages = numpy.round( pca.explained_variance_ratio_ * 100, 1 )

# plotting
figPCA = figure(figsize=(10, 8))

plot( pcDataFrame.loc[ metadata.sex == "Male", "pc1"], pcDataFrame.loc[ metadata.sex == "Male", "pc2" ], 'o', markersize=10, color="CornFlowerBlue", label="Males" )

plot( pcDataFrame.loc[ metadata.sex == "Female", "pc1"], pcDataFrame.loc[ metadata.sex == "Female", "pc2" ], 'o', markersize=10, color="FireBrick", label="Females" )

pyplot.title("PCA on normalized RNA levels", fontsize=20)

pyplot.xlabel("PC1 (" + str(pcentages[0]) + "%)", fontsize=16)
pyplot.ylabel("PC2 (" + str(pcentages[1]) + "%)", fontsize=16)
#pyplot.legend()
pyplot.text(x=-80, y=60, s="Males", fontsize=20, color="CornFlowerBlue")
pyplot.text(x=30, y=70, s="Females", fontsize=20, color="FireBrick")

#figPCA.savefig( os.path.join( project_dir, "figures", "pca.pdf") )  # optionally, save figure to a "figures" dir insider the project folder

show()
