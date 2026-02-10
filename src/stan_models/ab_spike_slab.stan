
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
    real<lower=0, upper=1> pi_prior;  // Prior inclusion probability
    real<lower=0> tau_slab;            // Slab variance (for included vars)
    real<lower=0> tau_spike;           // Spike variance (for excluded vars, small)
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    // Mixture probability (can be learned or fixed)
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
    // Likelihood
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
    // Donor priors
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(4, 1);
    sigma_chi ~ normal(0, 1); 
    
    // Spike-and-slab prior (marginalized)
    pi ~ beta(1, 1);  // or fix pi = pi_prior
    
    for (k in 1:p) {
        target += log_mix(pi,
                         normal_lpdf(beta[k] | 0, tau_slab),
                         normal_lpdf(beta[k] | 0, tau_spike));
    }
    
    // Random effects
    to_vector(beta_random) ~ normal(0, sigma_beta);
    sigma_beta_sq ~ inv_gamma(2, 1);  
}

generated quantities {
    vector[N] log_lik;
    vector[p] gamma;  // Posterior inclusion probabilities
    
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
    
    // Compute posterior inclusion probability for each coefficient
    for (k in 1:p) {
        real log_slab = log(pi) + normal_lpdf(beta[k] | 0, tau_slab);
        real log_spike = log1m(pi) + normal_lpdf(beta[k] | 0, tau_spike);
        gamma[k] = exp(log_slab - log_sum_exp(log_slab, log_spike));
    }
}

