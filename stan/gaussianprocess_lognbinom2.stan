functions {
    matrix multi_k_matrix(vector x, real[] params, matrix Kf, int M, int N, real deltadiag) {
    // sets up covariance matrix for multi-channel gaussian process
    matrix[M*N, M*N] K;
    int subl; // block indices
    int subk;

    for (l in 1:M) { // loop over channels (genes)
        real ella = params[l];
        subl = (l-1)*N;

        for (k in 1:M) {
            real ellb = params[k];
            subk = (k-1)*N;

            for (i in 1:N) { // loop over observation points
                K[subl+i,subk+i] = Kf[l,k];
                for (j in (i+1):N) {
                    K[subl+i,subk+j] = Kf[l,k] * exp( -pow( fabs( x[i] - x[j] ), 2)/( pow(ella,2) + pow(ellb,2) ) );
                    K[subl+j,subk+i] = K[subl+i,subk+j];
                }
            }
        }
    }
    return K + diag_matrix( rep_vector( deltadiag, M*N ) );
    }

    matrix assemble_Kf(real[] KfDiag, real[] KfTril, int M) {
    // assembles matrix of signal variance/covariance parameters from vector variable
        matrix[M, M] Kf;
        int suml = 1;
        for (l in 1:(M - 1)) {
            for (k in (l + 1):M) {
                Kf[l, k] = KfTril[suml];
                Kf[k, l] = Kf[l, k];
                suml = suml+1;
            }
            Kf[l, l] = KfDiag[l];
        }
        Kf[M,M] = KfDiag[M];
        return Kf;
    }

    vector multichannel_vector(real[] paramlist, int M, int N)
    {
    // sets up vector from list
    vector[M*N] paramVec;
    int subl;
    for (l in 1:M) {
        subl = (l-1)*N;
        for (k in 1:N) {
            paramVec[subl+k] = paramlist[l];
        }
    }
    return paramVec;
    }
}
data {
    int<lower=1> N;  // number of time points with observations
    int<lower=1> M; // number of channels
    int trilM; // number of off-diagonal signal covariances
    vector[N] gen; // time points (generations) with observations

    int y1a[M*N]; // first set of observations (sample replicates) in experimental replicate 1
    int y2a[M*N];
    int y1b[M*N];
    int y2b[M*N]; // second set of observations (sample replicates) in experimental replicate 2

    vector[M*N] sf1a;
    vector[M*N] sf2a;
    vector[M*N] sf1b;
    vector[M*N] sf2b;

    real meanPrior[M]; // mean value of data
    real stdPrior[M]; // standard deviation value of data

    real<lower=0> KfDiagMu[M]; // signal covariances prior mean (obtained form single-channel)
    real<lower=0> KfDiagSigma[M]; // signal covariances prior variance (root) (obtained form single-channel)

    real<lower=0> ellMu[M]; // bandwidth covariances prior mean (obtained form single-channel)
    real<lower=0> ellSigma[M]; // bandwidth covariances prior variance (root) (obtained form single-channel)
}
transformed data {
    real delta = 1e-8; // jitter term (for numerical stability)

    real mu[M] = meanPrior; // list of gaussian process means
    real stdPriorMax = max(stdPrior); // maximum standard deviation from `stdPrior`

    vector[M*N] logSizefactor1a = log(sf1a); // expression normalizing factors
    vector[M*N] logSizefactor2a = log(sf2a);
    vector[M*N] logSizefactor1b = log(sf1b);
    vector[M*N] logSizefactor2b = log(sf2b);
}
parameters {
    //real mu[M];

    real<lower=0> KfDiag[M];  // signal variance parameters for M channels
    real KfTril[trilM]; // signal covariances (off-diagonal blocks)

    real<lower=0> ell[M]; // bandwidth parameters

    vector[M*N] f1_til; // normal variates from standard independent gaussians
    vector[M*N] f2_til;

    real<lower=0> phi[M]; // inverse of square dispersion parameter
}
transformed parameters {
    real alpha[M]; // dispersion parameter
    for (l in 1:M) { alpha[l] = 1/sqrt(phi[l]); }
} // placing cholesky_decompose raised errors
model {
    vector[M*N] meanVector = multichannel_vector(mu, M, N); // vector of mean values
    vector[M*N] alphaVector = multichannel_vector(alpha, M, N); // vector of dispersion values

    matrix[M, M] Kf = assemble_Kf(KfDiag, KfTril, M);  // matrix of signal variances/covariances only
    matrix[M*N, M*N] K = multi_k_matrix(gen, ell, Kf, M, N, delta); // full gaussian process covariance matrix

    matrix[M*N, M*N] L = cholesky_decompose(K);

    vector[M*N] f1 = meanVector + L * f1_til; // model output for replicate 1
    vector[M*N] f2 = meanVector + L * f2_til;

    // priors
    ell ~ normal(ellMu, ellSigma); // gamma(ellShape, ellRate);
    KfDiag ~ normal(KfDiagMu, KfDiagSigma);  // gamma(KfDiagShape, KfDiagRate);

    KfTril ~ normal(0, stdPriorMax);

    f1_til ~ normal(0, 1);
    f2_til ~ normal(0, 1);

    phi ~ normal(0, 1);

    // likelihoods
    y1a ~ neg_binomial_2_log(f1 + logSizefactor1a, alphaVector);
    y1b ~ neg_binomial_2_log(f1 + logSizefactor1b, alphaVector);
    y2a ~ neg_binomial_2_log(f2 + logSizefactor2a, alphaVector);
    y2b ~ neg_binomial_2_log(f2 + logSizefactor2b, alphaVector);
}
generated quantities {}
