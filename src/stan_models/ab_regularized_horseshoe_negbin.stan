
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
    
    real<lower=0> p0;
    real mu_chi_mu;
    real<lower=0> mu_chi_sigma;
    real<lower=0> sigma_chi_sigma;
    real<lower=0> sigma_beta_sq_a;
    real<lower=0> sigma_beta_sq_b;
    real<lower=0> phi_y_rate;
    real<lower=0> phi_d_rate;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    vector<lower=0>[p] lambda;
    real<lower=0> tau;
    real<lower=0> c_sq;
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    vector<lower=0>[p] lambda_tilde;
    
    {
        real c = sqrt(c_sq);
        for (k in 1:p) {
            lambda_tilde[k] = sqrt(c_sq * square(lambda[k]) / 
                                   (c_sq + square(tau) * square(lambda[k])));
        }
    }
    
    for(i in 1:N) {
        mu[i] = alpha[idx_experiment[i]] * exp(row(X, i) * beta + 
                beta_random[idx_experiment[i], idx_experiment_replica[i]]);
    }
}

model {   
    Y ~ neg_binomial_2(rho_trasc .* mu, phi_Y);
    D ~ neg_binomial_2(rho_donor .* alpha[idx_donor_experiment], phi_D);
    
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0, sigma_chi_sigma); 
    
    lambda ~ cauchy(0, 1);
    tau ~ cauchy(0, p0 / (p - p0) * sigma_chi / sqrt(N));
    c_sq ~ inv_gamma(0.5, 0.5);
    
    for (k in 1:p) {
        beta[k] ~ normal(0, tau * lambda_tilde[k]);
    }
    
    to_vector(beta_random) ~ normal(0, sigma_beta);
    sigma_beta_sq ~ inv_gamma(sigma_beta_sq_a, sigma_beta_sq_b);
    
    phi_Y ~ exponential(phi_y_rate);
    phi_D ~ exponential(phi_d_rate);
}

generated quantities {
    vector[N] log_lik;
    array[N] int<lower=0> Y_rep;
    array[M] int<lower=0> D_rep;
    
    for (n in 1:N) {
        log_lik[n] = neg_binomial_2_lpmf(Y[n] | rho_trasc[n] * mu[n], phi_Y);
        Y_rep[n] = neg_binomial_2_rng(rho_trasc[n] * mu[n], phi_Y);
    }
    
    for (l in 1:M) {
        D_rep[l] = neg_binomial_2_rng(rho_donor[l] * alpha[idx_donor_experiment[l]], phi_D);
    }
}

