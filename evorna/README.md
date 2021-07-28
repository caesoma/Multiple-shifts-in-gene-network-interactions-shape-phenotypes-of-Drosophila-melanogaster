# Multiple shifts in gene network interactions shape phenotypes of _Drosophila melanogaster_ selected for long and short night sleep duration

This package provides functions to reproduce analyses and figures in Souto-Maior _et al_. bioRxiv **2021**:451943; [doi: 10.1101/2021.07.11.451943](https://doi.org/10.1101/2021.07.11.451943). See repo README for more detailed explanation and walkthrough.

- The `rnaseq` module provides functions to load, normalize, and filter RNA count data.

- The `phenotypic` module (package) provides the modules with functions to perform the analyses of phenotypic summaries and heritability and plot the results

- The `diffx` module (package) provides the modules and functions to set up and perform the hierarchical generalized linear model (GLM) analysis and plot the results

- The `correlation` module (package) provides simple functions to compute nonparametric (Spearman) correlations between several pairs of genes

- The `nonlinear` module (package) contains the `gaussian_processes` module and other functions to set up and run the Gaussian Process inference and analyses.
