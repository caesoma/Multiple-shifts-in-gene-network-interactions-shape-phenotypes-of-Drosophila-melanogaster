#!/usr/bin/python3

import os
import numpy, pandas

from evorna import estimate_size_factors

def count_data(data_dir, data_file):
    """ reads comma-separated count data with header and labels """
    data = pandas.read_csv( os.path.join(data_dir, data_file), sep=",", header=0, index_col=0 )
    return data


def check_labels(data_dir, meta_file, genic_file, inter_file, transpose=False):
    """ checks that indices in data and metadata match, returns genic, intergenic, and metadata files as they were read if no mismatch is found, otherwise reaises `ValueError`. `transpose` option returns transposed data frame """

    metadata = count_data(data_dir, meta_file)
    genic = count_data(data_dir, genic_file)
    intergenic = count_data(data_dir, inter_file)

    # check that metadata and data sample labels match
    if (genic.index != metadata.index).all():
        raise ValueError(">>> design and genic labels do not match: check data.")
    elif (intergenic.index != metadata.index).all():
        raise ValueError(">>> design and intergenic labels do not match: check data.")
    else:
        print(">>> design and data labels are a match.")

    if (transpose==True):
        return metadata, genic.T, intergenic.T
    else:
        return metadata, genic, intergenic


def pool_data(data_dir, meta_file, genic_file, inter_file, transpose=False):
    """ reads pooled data with checked labels and looks for a `pool` variables; if not found will build a consistent pool with two different labels labeling based on `sample_replicate` variable that may have more than two labels """
    metadata, genic, intergenic = check_labels(data_dir, meta_file, genic_file, inter_file)

    # define sets of design variables
    sexes = list(set(metadata.sex)); sexes.sort()
    selschemes = list(set(metadata.sel)); selschemes.sort()
    replicates = list(set(metadata.Rep)); replicates.sort()
    generations = list(set(metadata.generation)); generations.sort()

    # if 'sample pool' variable not in metadata, generate that from current coding so that each subset has two pools with same two numbers across all samples -- i.e. subsetting data by sampled pool returns half of the data
    if 'pool' not in metadata.columns:
        print(">>> sample pool label not defined, creating label from sample_replicate labels.")
        metadata['pool'] = int(9)
        for sexx in sexes:
            for sell in selschemes:
                for Repp in replicates:
                    for genn in generations:
                        subpools = metadata[ ( metadata.sex==sexx ) & ( metadata.sel == sell ) & ( metadata.Rep == Repp ) & ( metadata.generation==genn ) ].sample_replicate.values
                        if len(subpools) > 2:
                            print("sex:", sexx, ", sel:", sell, ", Rep:", Repp, ", generation:", genn)
                            raise ValueError("more than two sample replicates for data subset.")
                        else:
                            metadata.loc[ ( ( metadata.sex==sexx ) & ( metadata.sel == sell ) & ( metadata.Rep == Repp ) & ( metadata.generation==genn ) ), 'pool'] = [1,2]

    if (transpose==True):
        return metadata, genic.T, intergenic.T
    else:
        return metadata, genic, intergenic


def sizefactors_data(data_dir, meta_file, genic_file, inter_file, transpose=False):
    """ reads data and computes size factors using `estimate_size_factors` function translated from DESeq2 implementation """
    metadata, genic, intergenic = pool_data( data_dir, meta_file, genic_file, inter_file )

    sizeFactors = estimate_size_factors.deseq2( pandas.concat( [ genic, intergenic ], axis=1 ) )
    metadata['sizeFactors'] = sizeFactors

    if (transpose==True):
        return metadata, genic.T, intergenic.T
    else:
        return metadata, genic, intergenic


def normalized_data(data_dir, meta_file, genic_file, inter_file, transpose=False):

    metadata, genic, intergenic = sizefactors_data(data_dir, meta_file, genic_file, inter_file)
    sizeFactors = metadata['sizeFactors']

    normGenic = genic.div(sizeFactors, axis=0)
    normInter = intergenic.div(sizeFactors, axis=0)

    if (transpose==True):
        return metadata, normGenic.T, normInter.T
    else:
        return metadata, normGenic, normInter


def filtered_data(data_dir, meta_file, genic_file, inter_file, normalized=False, transpose=False):

    _, genicT, intergenicT = sizefactors_data(data_dir, meta_file, genic_file, inter_file, transpose=True)
    metadata, normGenicT, normInterT = normalized_data(data_dir, meta_file, genic_file, inter_file, transpose=True)

    # compute log2 of mean expression (aggregated over samples, "axis=1"), excluding genes with mean expression equal to zero
    l2rawgenic = numpy.log2( normGenicT[ normGenicT.mean(axis=1) > 0 ].mean( axis=1) )
    l2rawinter = numpy.log2( normInterT[ normInterT.mean(axis=1) > 0 ].mean( axis=1) )

    # define RNA expression cut-off as the 0.95 quantile of log intergenic mean expression
    exprCutoff = numpy.percentile(l2rawinter, 95.0)

    # filter genes with log2 of mean expression (mean computer over axis=1, all samples) with lower than the 0.95 quantile of intergenic regions
    filterGenicT = genicT[ ( normGenicT.mean( axis=1 ) > 0 ) & ( l2rawgenic > exprCutoff ) ] if (normalized==False) else normGenicT[ ( normGenicT.mean( axis=1 ) > 0 ) & ( l2rawgenic > exprCutoff ) ]

    filterGenic = filterGenicT.T if (normalized==False) else filterGenicT.div(metadata.sizeFactors, axis=1).T

    if (transpose==True):
        return metadata, filterGenic.T
    else:
        return metadata, filterGenic


def filtered_sex_data(data_dir, meta_file, genic_file, inter_file, sex, normalized=False, transpose=False):

    metadata, normGenicT, normInterT = normalized_data(data_dir, meta_file, genic_file, inter_file, transpose=True)
    _, genicT, _ = check_labels(data_dir, meta_file, genic_file, inter_file, transpose=True)

    # sex-specific cutoff filter...
    sexNormGenicT = normGenicT.T[ metadata.sex == sex ].T
    sexNormInterT = normInterT.T[ metadata.sex == sex ].T
    metaDataSex = metadata[ metadata.sex == sex  ]

    sexl2rawgenic = numpy.log2( sexNormGenicT[ sexNormGenicT.mean( axis=1 ) > 0 ].mean( axis=1 ) )
    sexl2rawinter = numpy.log2( sexNormInterT[ sexNormInterT.mean( axis=1 ) > 0 ].mean( axis=1 ) )

    sexxprCutoff = numpy.percentile( sexl2rawinter, 95.0 )

    filterSexGenicT = genicT.loc[ ( sexNormGenicT.mean( axis=1 ) > 0 ) & ( sexl2rawgenic > sexxprCutoff ), ( metadata.sex == sex ) ]

    filterGenic = filterSexGenicT.T if (normalized==False) else filterSexGenicT.div(metaDataSex.sizeFactors, axis=1).T

    if (transpose==True):
        return metaDataSex, filterGenic.T
    else:
        return metaDataSex, filterGenic


#@numba.jit
def check_axis_consistency(metaDataFrame):
    """ check generation values and ordering across subsets """
    print(">>> checking generation values and ordering:")
    sexes = list(set(metaDataFrame.sex));
    sexes.sort()
    selschemes = list(set(metaDataFrame.sel));
    selschemes.sort()
    replicates = list(set(metaDataFrame.Rep));
    replicates.sort()
    pools = list(set(metaDataFrame.pool));
    pools.sort()

    generations = list(set(metaDataFrame.generation));
    generations.sort()

    basegen = metaDataFrame[(metaDataFrame.sel == "control") & (metaDataFrame.Rep == 1) & (metaDataFrame.pool == 1)].generation.values

    print("gens:", numpy.array(generations))
    print("\nbase:", basegen)
    genflag = True
    for sell in selschemes:
        for Repp in replicates:
            for poo in pools:
                testgen = metaDataFrame[
                    (metaDataFrame.sel == sell) & (metaDataFrame.Rep == Repp) & (metaDataFrame.pool == poo)].generation.values
                if (testgen != basegen).all():
                    print(">>> generation values different across subsets: reorder samples")
                    genflag = False
                    del basegen
                    return pandas.DataFrame([])
                else:
                    print("curr:", testgen)
                return metaDataFrame
