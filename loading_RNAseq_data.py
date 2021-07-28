#!/usr/bin/python
#!/usr/bin/python3

"""
Plots results of gaussian process covariances for males and female flies
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Tutorial"

import os, sys, platform
import numpy, pandas
from scipy.stats import scoreatpercentile as scapl

import evorna.rnaseq as rnaseq


project_dir = os.getcwd() # assumes this script is being run from one level above the `evorna` package
data_dir = os.path.join( os.getcwd(), "data" )

genic_file = os.path.join( data_dir, "genic.csv" )
intergenic_file = os.path.join( data_dir, "intergenic.csv" )
metadata_file = os.path.join( data_dir, "metadata.csv" )
flybase_file = os.path.join( data_dir, "Flybase_ID_gene_info_111717.csv" )



# loads genic and intergenic count array, with genes as rows and samples as columns
print(">>> Loading raw data")

# loads count data creating a `pool` label for repeated time points within a single experiment (a.k.a. `sample_replicate`) -in this next line metadata is not assigned to a variable with proper name
_, genic, intergenic = rnaseq.pool_data(data_dir, metadata_file, genic_file, intergenic_file, transpose=False)

# loads count `pool_data` and computes size factors, metadata here contains both `pool` and `sizefactor` attributes
metadata0, genic, intergenic = rnaseq.sizefactors_data(data_dir, metadata_file, genic_file, intergenic_file)

# loads normalized data, metadata is the same as above. Transposition of data frame is false by default (i.e. if ommitted), so if needed it must be explicitly requested
_, normGenic, normIntergenic = rnaseq.normalized_data(data_dir, metadata_file, genic_file, intergenic_file, transpose=True)

# loads data with lowly expressed genes filtered out, default returns non-normalized data `normalized=True` option must be specified otherwise
_, filterNormGenic = rnaseq.filtered_data(data_dir, metadata_file, genic_file, intergenic_file, normalized=True)

# checks that generation labels are correctly sorted for all subsets, allowing their use with the same model
metadata = rnaseq.check_axis_consistency(metadata0)

# if filtering is not used single sex data can be obtained by simply subsetting the data based on `sex` variable, otherwise the `data_sex_filtered` function does sexes separate filtering (disregards expression data from other sex).
metaDataFemale0, filterFemaleGenic = rnaseq.filtered_sex_data(data_dir, metadata_file, genic_file, intergenic_file, "Female", normalized=False)
metaDataFemale = rnaseq.check_axis_consistency(metaDataFemale0)

metaDataMale0, filterMaleGenic = rnaseq.filtered_sex_data(data_dir, metadata_file, genic_file, intergenic_file, "Male")
metaDataMale = rnaseq.check_axis_consistency(metaDataMale0)
