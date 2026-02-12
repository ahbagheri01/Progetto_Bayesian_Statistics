
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
    
    real mu_alpha_0;
    real<lower=0> sigma_alpha_0;
    real<lower=0> sigma_b;
    real<lower=0> phi_y_rate;
    real<lower=0> phi_d_rate;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta;
    real<lower=0> tau;
    real mu_chi;
    real<lower=0> sigma_chi;
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
}

transformed parameters {
    vector[N] mu;
    for(i in 1:N) {
        mu[i] = alpha[idx_experiment[i]] * exp(row(X, i) * beta + beta_random[idx_experiment[i], idx_experiment_replica[i]]);
    }
}

model {   
    for (s in 1:N) {
        Y[s] ~ neg_binomial_2(rho_trasc[s] * mu[s], phi_Y);
    }

    for (l in 1:M) {
        D[l] ~ neg_binomial_2(rho_donor[l] * alpha[idx_donor_experiment[l]], phi_D);
    }

    for (m in 1:I) {
        log(alpha[m]) ~ normal(mu_chi, sigma_chi);
    }
    mu_chi ~ normal(mu_alpha_0, sigma_alpha_0);
    sigma_chi ~ normal(0, sigma_alpha_0); 
    
    for (k in 1:p) {
        beta[k] ~ normal(0.0, tau);
    }
    tau ~ normal(0, sigma_b); 
    
    for (i in 1:I) {    
        for (j in 1:J) {
            beta_random[i,j] ~ normal(0, sigma_beta);
        }
    }
    sigma_beta ~ normal(0, sigma_b);
    
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

