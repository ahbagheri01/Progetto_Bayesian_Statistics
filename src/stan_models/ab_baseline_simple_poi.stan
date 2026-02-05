
data {
    int<lower=0> N; 
    int<lower=0> M; 
    int<lower=0> p; 
    int<lower=0> I;
    int<lower=0> J;
    array[N] int<lower=0> Y;
    array[M] int<lower=0> D;
    matrix[N, p] X;
    vector[N] rho_trasc;
    vector[M] rho_donor;
    array[N] int idx_experiment;
    array[N] int idx_experiment_replica;
    array[M] int idx_donor_experiment;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real<lower=0> tau_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
}

transformed parameters {
    vector[N] mu;
    for(i in 1:N) {
      mu[i] = alpha[idx_experiment[i]]* exp(row(X, i) * beta + beta_random[idx_experiment[i], idx_experiment_replica[i]]);
    }
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> tau = sqrt(tau_sq);
}

model {   
    for (s in 1:N) {
        Y[s] ~ poisson(rho_trasc[s] * mu[s]);  
    }

    for (l in 1:M){
        D[l] ~ poisson(rho_donor[l] * alpha[idx_donor_experiment[l]]);
    }

    for (m in 1:I) {
        log(alpha[m])~normal(mu_chi, sigma_chi);
    }
    mu_chi ~ normal(4,1);
    sigma_chi ~ normal(0,1); 
    
    for (k in 1:p) {
        beta[k] ~ normal(0.0, tau);
    }

    tau_sq ~ inv_gamma(2, 1); 
    
    for (i in 1:I) {    
        for (j in 1:J) {
            beta_random[i,j] ~ normal(0, sigma_beta);
        }
    }
    sigma_beta_sq ~ inv_gamma(2, 1);  
}

generated quantities {
  vector[N] log_lik;
  for (j in 1:N) {
    log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j]* mu[j]);
  }
}

