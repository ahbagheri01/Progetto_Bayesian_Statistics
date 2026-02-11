
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
    
    // Hyperparameters for priors
    real mu_chi_mu;
    real<lower=0> mu_chi_sigma;
    real<lower=0> sigma_chi_sigma;
    real<lower=0> tau_sq_a;  
    real<lower=0> tau_sq_b;
    real<lower=0> sigma_beta_sq_a;  
    real<lower=0> sigma_beta_sq_b;
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
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> tau = sqrt(tau_sq);
    for(i in 1:N) {
      mu[i] = alpha[idx_experiment[i]]* exp(row(X, i) * beta + beta_random[idx_experiment[i], idx_experiment_replica[i]]);
    }
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
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0,sigma_chi_sigma); 
    
    for (k in 1:p) {
        beta[k] ~ normal(0.0, tau);
    }

    tau_sq ~ inv_gamma(tau_sq_a, tau_sq_b); 
    
    for (i in 1:I) {    
        for (j in 1:J) {
            beta_random[i,j] ~ normal(0, sigma_beta);
        }
    }
    sigma_beta_sq ~ inv_gamma(sigma_beta_sq_a, sigma_beta_sq_b);  
}

generated quantities {
  vector[N] log_lik;
  array[N] int<lower=0> Y_rep;

  for (j in 1:N) {
    log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j]* mu[j]);
  }

  for (n in 1:N) {
    Y_rep[n] = poisson_rng(rho_trasc[n] * mu[n]);
  } 
}

