# Multiple shifts in gene network interactions shape phenotypes of _Drosophila melanogaster_ selected for long and short night sleep duration

This repo provides functions and walkthrough scripts to reproduce analyses and figures in Souto-Maior _et al_. bioRxiv **2021**:451943; [doi: 10.1101/2021.07.11.451943](https://doi.org/10.1101/2021.07.11.451943).

This repo contains the `evorna` package with modules and functions to reproduce all Python analyses in the manuscript (see package README and docstrings for more details on moduels and functions). It also contains walkthrough scripts to use those functions to reproduce the figures generated in Python as seen in the manuscript from scratch (i.e. from count data). Additional folders in the repo (outside of the `evorna` folder) contain the Stan model code (`/stan`) and some "default" locations for the data, figures, and tables (as well as some package installation related files: `dist` and `evorna.egg-info`).

The Python scripts in the repo root (e.g. `compute_and_plot_PCA.py`) can be run directly from this location without need to install the package, as long as the dependencies in `pyproject.toml` are satisfied -- most of them are very common scientific computing packages (e.g. numpy, scipy, matplotlib), the essential packages are interface packages to the Stan software -- `cmdstanpy` and/or `pystan` -- needed to run both GLM and Gaussian Process inference. Otherwise, the `evorna` package can be installed using `pip` and scripts can be run from anywhere, just making sure the `project_dir`, `data_dir`, and `results_dir` exist and contain the subfolders or files as applicable.

CmdStanPy is required to run the Gaussian Process analyses, and therefore [CmdStan](https://github.com/stan-dev/cmdstan) needs to be installed separately so the `cmdstan_path` variable and external model compilation can be done -- it is also the best way to parallelize the 85680 MCMC runs; it is in principle possible to modify the scripts to use PyStan, but unless you have a good reason to try it it's not recommended. For the GLM analysis two modules are provided, using either CmdStanPy (`hierarchical_glm`) or Pystan (`hierarchical_glm_pystan`) -- the latter does have the advantage of being self-contained within Python and not requiring external programs, but parallelization is not straightforward (template for parallelization is provided but it doesn't work in its current form).

The walkthrough scripts included in the repo are the following:

- `phenotypic_analyses.py`: computes the traits' summaries and plots their values along time (Figure 1A, B and Figure S2); computes heritability measures and plots the regression on the cumulated differentials (Figure 1C,D,E);

- `loading_RNAseq_data.py`: examples of loading RNA-seq data as raw, normalized, or filtered transcript counts;

- `compute_and_plot_PCA.py`: computes PCA on normalized RNA-seq data and labels them by sex (Figure S1);

- `hierarchical_GLM_walkthrough.py`: runs GLM analysis on a few genes and produces the plot in Figure 2;

- `compute_spearman_correlations.py`: Computes nonparametric (Spearman) correlations between a subset of genes (85 genes, 3570 pairs) as produces data frame wih results (part of Table S10);

- `multi-channel_Gaussian_Process_walkthrough.py`: runs single-channel GP analysis on two individual genes and multi-channel GP inference on both genes together -- produces figure 3 of the manuscript and an extremely reduced version of Figure 4 (3570 similar runs are needed to produce the whole figure, and parallelizing the runs outside of Python is recommended).


If there are any questions or trouble feel free to contact the authors (e-mail on publication) about it.
