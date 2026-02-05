
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
    
    // R2 prior parameters
    real<lower=0, upper=1> R2_mean;
    real<lower=0> R2_prec;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    // R2-D2 parameters
    simplex[p] psi;
    real<lower=0, upper=1> R2;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> sigma_beta_r2;
    vector[p] lambda_r2d2;
    
    // R2-D2 shrinkage
    sigma_beta_r2 = sqrt(R2 / (1 - R2));
    for (k in 1:p) {
        lambda_r2d2[k] = sigma_beta_r2 * sqrt(psi[k]);
    }
    
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
    
    // R2-D2 prior
    R2 ~ beta(R2_mean * R2_prec, (1 - R2_mean) * R2_prec);
    psi ~ dirichlet(rep_vector(1.0, p));
    
    for (k in 1:p) {
        beta[k] ~ normal(0, lambda_r2d2[k]);
    }
    
    // Random effects
    to_vector(beta_random) ~ normal(0, sigma_beta);
    sigma_beta_sq ~ inv_gamma(2, 1);  
}

generated quantities {
    vector[N] log_lik;
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
}

