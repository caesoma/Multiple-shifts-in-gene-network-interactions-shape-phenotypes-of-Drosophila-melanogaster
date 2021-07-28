functions {
  vector linear_model(real intercept1, real intercept2, real bShort1, real bShort2, real bLong1, real bLong2, real gen, real bShortGen1, real bShortGen2, real bLongGen1, real bLongGen2, matrix designMatrix, int N, int P) {
        // full linear model (has all coefficients in the experimental design)

        vector[P] coefficients = [intercept1, intercept2, bShort1, bShort2, bLong1, bLong2,
     gen, bShortGen1, bShortGen2, bLongGen1, bLongGen2]'; // creates vector with coefficients/model parameters

        vector[N] logqj = designMatrix * coefficients;

        return logqj;
  }
}
data {
    int<lower=1> N;
    int<lower=1> P;

    matrix[N,P] designMatrix;  // must follow parameter order given by `linear_model` function

    int y[N];
    vector[N] sfs;

    real printercept; // mean of data at first time point (generation zero)
    //real stdcept;
}
transformed data {}
parameters {
    // 'hyper-parameters'

    // group-level intercept parameters common to all schemes (including controls)
    real muControl;
    real<lower=0> sigmaControl;

    // group-level intercept parameters for selected schemes (excludes controls)
    real muShort;
    real<lower=0> sigmaShort;
    real muLong;
    real<lower=0> sigmaLong;

    // group-level interaction parameters for selected schemes
    real muShortGen;
    real<lower=0> sigmaShortGen;
    real muLongGen;
    real<lower=0> sigmaLongGen;

    // 'hypo-parameters' (i.e. 'parameters')
    real bIntercept1; // intercept parameter for all Replicate 1 schemes
    real bIntercept2;

    real bShort1; // intercept parameter for Replicate 1 Short sleepers
    real bShort2;
    real bLong1;
    real bLong2;

    real gen; // slope along generations for all schemes

    real bShortGen1; // slope along generations for Replicate 1 Short sleepers
    real bShortGen2;
    real bLongGen1;
    real bLongGen2;

    real<lower=0> alpha; // Negative Binomial ("second parameterization") dispersion parameter
}
transformed parameters {
    // full model prediction ('ln' link function)
    vector[N] full = exp( linear_model( bIntercept1, bIntercept2, bShort1, bShort2, bLong1, bLong2, gen, bShortGen1, bShortGen2, bLongGen1, bLongGen2,  designMatrix, N, P ) );
}
model {
    // priors
    muShort ~ normal(0,1);
    sigmaShort ~ cauchy(0,5);
    muControl ~ normal(printercept,1);
    sigmaControl ~ cauchy(0,5);
    muLong ~ normal(0,1);
    sigmaLong ~ cauchy(0,5);

    muShortGen ~ normal(0,1);;
    sigmaShortGen ~ cauchy(0,5);
    muLongGen ~ normal(0,1);;
    sigmaLongGen ~ cauchy(0,5);

    bIntercept1 ~ normal(muControl,sigmaControl);
    bIntercept2 ~ normal(muControl,sigmaControl);

    bShort1 ~ normal(muShort,sigmaShort);
    bShort2 ~ normal(muShort,sigmaShort);
    bLong1 ~ normal(muLong,sigmaLong);
    bLong2 ~ normal(muLong,sigmaLong);

    gen ~ normal(0,2);;

    bShortGen1 ~ normal(muShortGen,sigmaShortGen);
    bShortGen2 ~ normal(muShortGen,sigmaShortGen);
    bLongGen1 ~ normal(muLongGen,sigmaLongGen);
    bLongGen2 ~ normal(muLongGen,sigmaLongGen);

    alpha ~ uniform(0, 1e9);

    // likelihood
    y ~ neg_binomial_2( ( full .* sfs ), alpha );
}
generated quantities {}
