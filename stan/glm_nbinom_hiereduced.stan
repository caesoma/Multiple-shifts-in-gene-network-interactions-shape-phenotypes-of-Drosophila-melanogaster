functions {
  vector reduced_model(real intercept1, real intercept2, real bShort1, real bShort2, real bLong1, real bLong2, real gen, matrix reducedDesign, int N, int R) {
      // reduced linear model (excludes interaction coefficients from full model)

        vector[R] coefficients = [intercept1, intercept2, bShort1, bShort2, bLong1, bLong2,
     gen]';

        vector[N] logqj = reducedDesign * coefficients;

        return logqj;
  }
}
data {
    int<lower=1> N;
    int<lower=1> R;

    matrix[N,R] reducedMatrix; // design matrix matching order from `reduced_model` function

    int y[N];
    vector[N] sfs;

    real printercept;
    //real stdcept;
}
transformed data {}
parameters {
    // 'hyper-parameters'
    real muShort;
    real<lower=0> sigmaShort;
    real muControl;
    real<lower=0> sigmaControl;
    real muLong;
    real<lower=0> sigmaLong;

    // 'parameters'
    real bIntercept1;
    real bIntercept2;

    real bShort1;
    real bShort2;
    real bLong1;
    real bLong2;

    real gen;

    real<lower=0> alpha;
}
transformed parameters {
    vector[N] reduced = exp( reduced_model( bIntercept1, bIntercept2, bShort1, bShort2, bLong1, bLong2, gen, reducedMatrix, N, R ) );
}
model {
    // see glm_nbinom_hierarchical stan model for detailed description of model and all other blocks
    muShort ~ normal(0,1);
    sigmaShort ~ cauchy(0,5);
    muControl ~ normal(printercept,1);
    sigmaControl ~ cauchy(0,5);
    muLong ~ normal(0,1);
    sigmaLong ~ cauchy(0,5);

    bIntercept1 ~ normal(muControl,sigmaControl);
    bIntercept2 ~ normal(muControl,sigmaControl);

    bShort1 ~ normal(muShort,sigmaShort);
    bShort2 ~ normal(muShort,sigmaShort);
    bLong1 ~ normal(muLong,sigmaLong);
    bLong2 ~ normal(muLong,sigmaLong);

    gen ~ normal(0,2);;

    alpha ~ uniform(0, 1e9);

    y ~ neg_binomial_2((reduced .* sfs), alpha);
}
generated quantities {}
