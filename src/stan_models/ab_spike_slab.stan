
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
    
    // Spike-and-slab parameters
    real<lower=0, upper=1> pi_prior;
    real<lower=0> tau_slab;
    real<lower=0> tau_spike;
    
    // Hyperparameters for priors
    real mu_chi_mu;
    real<lower=0> mu_chi_sigma;
    real<lower=0> sigma_chi_sigma;
    real<lower=0> sigma_beta_sq_a;
    real<lower=0> sigma_beta_sq_b;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    real<lower=0, upper=1> pi;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    
    for(i in 1:N) {
      mu[i] = alpha[idx_experiment[i]] * exp(row(X, i) * beta + 
              beta_random[idx_experiment[i], idx_experiment_replica[i]]);
    }
}

model {   
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0, sigma_chi_sigma); 
    
    pi ~ beta(1, 1);
    
    for (k in 1:p) {
        target += log_mix(pi,
                         normal_lpdf(beta[k] | 0, tau_slab),
                         normal_lpdf(beta[k] | 0, tau_spike));
    }
    
    to_vector(beta_random) ~ normal(0, sigma_beta);
    sigma_beta_sq ~ inv_gamma(sigma_beta_sq_a, sigma_beta_sq_b);  
}

generated quantities {
    vector[N] log_lik;
    vector[p] gamma;
    array[N] int<lower=0> Y_rep; 
    
    for (n in 1:N) {
        Y_rep[n] = poisson_rng(rho_trasc[n] * mu[n]);
    }  
    
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
    
    for (k in 1:p) {
        real log_slab = log(pi) + normal_lpdf(beta[k] | 0, tau_slab);
        real log_spike = log1m(pi) + normal_lpdf(beta[k] | 0, tau_spike);
        gamma[k] = exp(log_slab - log_sum_exp(log_slab, log_spike));
    }
}

