functions {
  vector reduced_model(real intercept1, real intercept2, real gen, matrix tinyDesign, int N, int T) {
        vector[T] coefficients = [intercept1, intercept2, gen]';

        vector[N] logqj = tinyDesign * coefficients;

        return logqj;
  }
}
data {
    int<lower=1> N;
    int<lower=1> T;

    matrix[N,T] tinyMatrix;

    int y[N];
    vector[N] sfs;

    real printercept;
    //real stdcept;
}
transformed data {}
parameters {
    // 'hyper-parameters'
    real muControl;
    real<lower=0> sigmaControl;

    // 'hypo-parameters'
    real bIntercept1;
    real bIntercept2;

    real gen;

    real<lower=0> alpha;
}
transformed parameters {
    vector[N] tiny = exp(reduced_model(bIntercept1, bIntercept2, gen, tinyMatrix, N, T));
}
model {
    muControl ~ normal(printercept,1);
    sigmaControl ~ cauchy(0,5);

    bIntercept1 ~ normal(muControl,sigmaControl);
    bIntercept2 ~ normal(muControl,sigmaControl);

    gen ~ normal(0,2);;

    alpha ~ uniform(0, 1e9);

    y ~ neg_binomial_2((tiny .* sfs), alpha);
}
generated quantities {}
