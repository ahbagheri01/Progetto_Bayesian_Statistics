import os
from cmdstanpy import CmdStanModel
from tensorflow_probability.substrates import numpy as tfp
tfd = tfp.distributions

import os
from pathlib import Path

current = Path.cwd()

if current.name == 'Progetto_Bayesian_Statistics':
    print(f"Already in project root: {current}")
else:
    for parent in [current] + list(current.parents):
        if parent.name == 'Progetto_Bayesian_Statistics':
            os.chdir(parent)
            print(f"Changed to: {parent}")
            break
    else:
        raise FileNotFoundError("Progetto_Bayesian_Statistics folder not found")

assert Path('src').exists(), "src/ folder not found"
print(f"Current directory: {os.getcwd()}")
print(f"Contents: {os.listdir()}")

STAN_PATH = "./src/stan_models/"
if not os.path.exists(STAN_PATH):
    os.mkdir(STAN_PATH)
    
import cmdstanpy
print(cmdstanpy.cmdstan_path())

glm1_pascal = """
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

    // Hyperparameters for dispersion priors (optional but nice)
    real<lower=0> phi_y_rate;
    real<lower=0> phi_d_rate;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real<lower=0> tau_sq;
    real mu_chi;
    real<lower=0> sigma_chi;

    // Pascal / Negative Binomial dispersion (shape)
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> tau = sqrt(tau_sq);

    for (i in 1:N) {
        mu[i] = alpha[idx_experiment[i]] *
                exp(row(X, i) * beta + beta_random[idx_experiment[i], idx_experiment_replica[i]]);
    }
}

model {   
    // Likelihood: Pascal / Negative Binomial (NB2)
    for (s in 1:N) {
        Y[s] ~ neg_binomial_2(rho_trasc[s] * mu[s], phi_Y);
    }

    for (l in 1:M) {
        D[l] ~ neg_binomial_2(rho_donor[l] * alpha[idx_donor_experiment[l]], phi_D);
    }

    // Priors (same as your Poisson baseline)
    for (m in 1:I) {
        log(alpha[m]) ~ normal(mu_chi, sigma_chi);
    }
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0, sigma_chi_sigma); 
    
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

    // Dispersion priors (weakly-informative; exponential is a safe baseline)
    phi_Y ~ exponential(phi_y_rate);
    phi_D ~ exponential(phi_d_rate);
}

generated quantities {
  vector[N] log_lik;
  array[N] int<lower=0> Y_rep;

  for (n in 1:N) {
    real mean_n = rho_trasc[n] * mu[n];
    log_lik[n] = neg_binomial_2_lpmf(Y[n] | mean_n, phi_Y);
    Y_rep[n]   = neg_binomial_2_rng(mean_n, phi_Y);
  }
}
"""
stan_file = f"{STAN_PATH}/ab_neg_binomial.stan"
with open(stan_file, "w") as f:
    print(glm1_pascal, file=f)
glm = CmdStanModel(stan_file=stan_file)




import os
from cmdstanpy import CmdStanModel
from pathlib import Path

current = Path.cwd()
if current.name == 'Progetto_Bayesian_Statistics':
    print(f"Already in project root: {current}")
else:
    for parent in [current] + list(current.parents):
        if parent.name == 'Progetto_Bayesian_Statistics':
            os.chdir(parent)
            print(f"Changed to: {parent}")
            break
    else:
        raise FileNotFoundError("Progetto_Bayesian_Statistics folder not found")

assert Path('src').exists(), "src/ folder not found"
print(f"Current directory: {os.getcwd()}")

STAN_PATH = "./src/stan_models/"
if not os.path.exists(STAN_PATH):
    os.mkdir(STAN_PATH)

# MODEL 1: Baseline Negative Binomial
glm1_nb = """
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
    
    real mu_chi_mu;
    real<lower=0> mu_chi_sigma;
    real<lower=0> sigma_chi_sigma;
    real<lower=0> tau_sq_a;  
    real<lower=0> tau_sq_b;
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
    real<lower=0> tau_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> tau = sqrt(tau_sq);
    
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
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0, sigma_chi_sigma); 
    
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
"""

stan_file = f"{STAN_PATH}/ab_baseline_negbin.stan"
with open(stan_file, "w") as f:
    print(glm1_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 2: Regularized Horseshoe NB
glm_horseshoe_reg_nb = """
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
"""

stan_file = f"{STAN_PATH}/ab_regularized_horseshoe_negbin.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_reg_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 3: Horseshoe NB
glm_horseshoe_nb = """
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
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
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
    Y ~ neg_binomial_2(rho_trasc .* mu, phi_Y);
    D ~ neg_binomial_2(rho_donor .* alpha[idx_donor_experiment], phi_D);
    
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0, sigma_chi_sigma); 
    
    lambda ~ cauchy(0, 1);
    tau ~ cauchy(0, 1);
    
    for (k in 1:p) {
        beta[k] ~ normal(0, tau * lambda[k]);
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
"""

stan_file = f"{STAN_PATH}/ab_horseshoe_negbin.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 4: R2D2 NB
glm_r2d2_nb = """
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
    
    real<lower=0, upper=1> R2_mean;
    real<lower=0> R2_prec;
    real<lower=0> cons_D2;
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
    real<lower=0> sigma;  
    simplex[p] phi;       
    real<lower=0, upper=1> R2;
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> omega;
    vector<lower=0>[p] lambda;
    
    omega = sqrt(R2 / (1 - R2));
    for (k in 1:p) {
        lambda[k] = sqrt(phi[k]) * omega;
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
    
    sigma ~ student_t(3, 0, 2.5);
    R2 ~ beta(R2_mean * R2_prec, (1 - R2_mean) * R2_prec);
    phi ~ dirichlet(rep_vector(cons_D2, p));
    
    for (k in 1:p) {
        beta[k] ~ normal(0, sigma * lambda[k]);
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
"""

stan_file = f"{STAN_PATH}/ab_r2d2_negbin.stan"
with open(stan_file, "w") as f:
    print(glm_r2d2_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 5: LASSO NB
glm_lasso_nb = """
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
    real<lower=0> lambda_lasso;
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
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
    Y ~ neg_binomial_2(rho_trasc .* mu, phi_Y);
    D ~ neg_binomial_2(rho_donor .* alpha[idx_donor_experiment], phi_D);
    
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(mu_chi_mu, mu_chi_sigma);
    sigma_chi ~ normal(0, sigma_chi_sigma); 
    
    lambda_lasso ~ cauchy(0, 1);
    beta ~ double_exponential(0, 1.0 / lambda_lasso);
    
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
"""

stan_file = f"{STAN_PATH}/ab_lasso_negbin.stan"
with open(stan_file, "w") as f:
    print(glm_lasso_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 6: Spike-and-Slab NB
glm_spike_slab_nb = """
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
    
    real<lower=0, upper=1> pi_prior;
    real<lower=0> tau_slab;
    real<lower=0> tau_spike;
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
    real<lower=0, upper=1> pi;
    real<lower=0> phi_Y;
    real<lower=0> phi_D;
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
    Y ~ neg_binomial_2(rho_trasc .* mu, phi_Y);
    D ~ neg_binomial_2(rho_donor .* alpha[idx_donor_experiment], phi_D);
    
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
    
    phi_Y ~ exponential(phi_y_rate);
    phi_D ~ exponential(phi_d_rate);
}

generated quantities {
    vector[N] log_lik;
    vector[p] gamma;
    array[N] int<lower=0> Y_rep;
    array[M] int<lower=0> D_rep;
    
    for (n in 1:N) {
        log_lik[n] = neg_binomial_2_lpmf(Y[n] | rho_trasc[n] * mu[n], phi_Y);
        Y_rep[n] = neg_binomial_2_rng(rho_trasc[n] * mu[n], phi_Y);
    }
    
    for (l in 1:M) {
        D_rep[l] = neg_binomial_2_rng(rho_donor[l] * alpha[idx_donor_experiment[l]], phi_D);
    }
    
    for (k in 1:p) {
        real log_slab = log(pi) + normal_lpdf(beta[k] | 0, tau_slab);
        real log_spike = log1m(pi) + normal_lpdf(beta[k] | 0, tau_spike);
        gamma[k] = exp(log_slab - log_sum_exp(log_slab, log_spike));
    }
}
"""

stan_file = f"{STAN_PATH}/ab_spike_slab_negbin.stan"
with open(stan_file, "w") as f:
    print(glm_spike_slab_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)


# MODEL 9: Half-Normal NB
glm_normal_nb = """
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
"""

stan_file = f"{STAN_PATH}/sr_halfnormal_negbin.stan"
with open(stan_file, "w") as f:
    print(glm_normal_nb, file=f)
glm = CmdStanModel(stan_file=stan_file)

print(f"\n✓ All Negative Binomial Stan models saved to {STAN_PATH}")
print(f"  - ab_baseline_negbin.stan")
print(f"  - ab_regularized_horseshoe_negbin.stan")
print(f"  - ab_horseshoe_negbin.stan")
print(f"  - ab_r2d2_negbin.stan")
print(f"  - ab_lasso_negbin.stan")
print(f"  - ab_spike_slab_negbin.stan")
print(f"  - sr_invgamma_negbin.stan")
print(f"  - sr_tstudent_negbin.stan")
print(f"  - sr_halfnormal_negbin.stan")