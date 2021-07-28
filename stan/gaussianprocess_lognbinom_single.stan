functions {
    matrix k_matrix(vector x, real ell, real sigmaf2, int N, real deltadiag) {
    matrix[N, N] K;

    for (i in 1:N) {
        K[i,i] = sigmaf2;
        for (j in (i+1):N) {
            K[i,j] = sigmaf2*exp( -pow( fabs( x[i] - x[j] ), 2 )/ ( 2 * pow(ell, 2) ) ) ;
            K[j,i] = K[i,j];
                }
            }
    return K + diag_matrix(rep_vector(deltadiag, N));
    }


}
data {
    int<lower=1> N;
    vector[N] gen;

    int y1a[N];
    int y2a[N];
    int y1b[N];
    int y2b[N];

    vector[N] sf1a;
    vector[N] sf2a;
    vector[N] sf1b;
    vector[N] sf2b;

    real meanPrior;
    real stdPrior;
}
transformed data {
    real delta = 1e-8;

    real mu = meanPrior;

    vector[N] logSizefactor1a = log(sf1a);
    vector[N] logSizefactor2a = log(sf2a);
    vector[N] logSizefactor1b = log(sf1b);
    vector[N] logSizefactor2b = log(sf2b);
}
parameters {
    //real mu;

    real<lower=0> KfDiag;

    real<lower=0> ell;

    vector[N] f1_til;
    vector[N] f2_til;

    real<lower=0> phi;
}
transformed parameters {
    real alpha  = 1/sqrt(phi);
} // placing cholesky_decompose here keeps raising errors
model {
    vector[N] meanVector = rep_vector(mu, N);
    vector[N] alphaVector = rep_vector(alpha, N);

    matrix[N, N] K = k_matrix(gen, ell, KfDiag, N, delta);

    matrix[N, N] L = cholesky_decompose(K);

    vector[N] f1 = meanVector + L * f1_til;
    vector[N] f2 = meanVector + L * f2_til;

    // mu ~ normal(meanPrior,stdPrior);
    ell ~ gamma(4, 4);
    KfDiag ~ normal(0, stdPrior);

    f1_til ~ normal(0, 1);
    f2_til ~ normal(0, 1);

    phi ~ normal(0, 1);

    y1a ~ neg_binomial_2_log(f1 + logSizefactor1a, alphaVector);
    y1b ~ neg_binomial_2_log(f1 + logSizefactor1b, alphaVector);
    y2a ~ neg_binomial_2_log(f2 + logSizefactor2a, alphaVector);
    y2b ~ neg_binomial_2_log(f2 + logSizefactor2b, alphaVector);
}
generated quantities {}
