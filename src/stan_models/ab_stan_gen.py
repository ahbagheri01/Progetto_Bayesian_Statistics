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

# MODEL 1: Baseline (already correct)
glm1 = """
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
"""

stan_file = f"{STAN_PATH}/ab_baseline_simple_poi.stan"
with open(stan_file, "w") as f:
    print(glm1, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 2: Regularized Horseshoe
glm_horseshoe_reg = """
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
    
    // Expected number of relevant variables
    real<lower=0> p0;
    
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
    
    vector<lower=0>[p] lambda;
    real<lower=0> tau;
    real<lower=0> c_sq;
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
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
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
}

generated quantities {
    vector[N] log_lik;
    array[N] int<lower=0> Y_rep; 
    
    for (n in 1:N) {
        Y_rep[n] = poisson_rng(rho_trasc[n] * mu[n]);
    }  
    
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
    

}
"""

stan_file = f"{STAN_PATH}/ab_regularized_horseshoe.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_reg, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 3: Horseshoe
glm_horseshoe = """
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
    
    vector<lower=0>[p] lambda;
    real<lower=0> tau;
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
    
    lambda ~ cauchy(0, 1);
    tau ~ cauchy(0, 1);
    
    for (k in 1:p) {
        beta[k] ~ normal(0, tau * lambda[k]);
    }
    
    to_vector(beta_random) ~ normal(0, sigma_beta);
    sigma_beta_sq ~ inv_gamma(sigma_beta_sq_a, sigma_beta_sq_b);  
}

generated quantities {
    vector[N] log_lik;
    array[N] int<lower=0> Y_rep; 
    
    for (n in 1:N) {
        Y_rep[n] = poisson_rng(rho_trasc[n] * mu[n]);
    }  
    
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
}
"""

stan_file = f"{STAN_PATH}/ab_horseshoe.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 4: R2D2
glm_r2d2 = """
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
    
    // R2-D2 prior parameters
    real<lower=0, upper=1> R2_mean;
    real<lower=0> R2_prec;
    real<lower=0> cons_D2;
    
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
    
    real<lower=0> sigma;  
    simplex[p] phi;       
    real<lower=0, upper=1> R2;
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
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
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
}

generated quantities {
    vector[N] log_lik;
    array[N] int<lower=0> Y_rep; 
    
    for (n in 1:N) {
        Y_rep[n] = poisson_rng(rho_trasc[n] * mu[n]);
    }  
    
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
}
"""

stan_file = f"{STAN_PATH}/ab_r2d2.stan"
with open(stan_file, "w") as f:
    print(glm_r2d2, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 5: LASSO
glm_lasso = """
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
    
    real<lower=0> lambda_lasso;
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
    
    lambda_lasso ~ cauchy(0, 1);
    beta ~ double_exponential(0, 1.0 / lambda_lasso);
    
    to_vector(beta_random) ~ normal(0, sigma_beta);
    sigma_beta_sq ~ inv_gamma(sigma_beta_sq_a, sigma_beta_sq_b);  
}

generated quantities {
    vector[N] log_lik;
    array[N] int<lower=0> Y_rep; 
    
    for (n in 1:N) {
        Y_rep[n] = poisson_rng(rho_trasc[n] * mu[n]);
    }  
    for (j in 1:N) {
        log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j] * mu[j]);
    }
}
"""

stan_file = f"{STAN_PATH}/ab_lasso.stan"
with open(stan_file, "w") as f:
    print(glm_lasso, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 6: Spike-and-Slab
glm_spike_slab = """
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
"""

stan_file = f"{STAN_PATH}/ab_spike_slab.stan"
with open(stan_file, "w") as f:
    print(glm_spike_slab, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 7: Inverse Gamma
glm_invgamma = """
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
    real mu_alpha_0;
    real<lower=0> sigma_alpha_0;
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
    mu_chi ~ normal(mu_alpha_0, sigma_alpha_0);
    sigma_chi ~ normal(0, sigma_alpha_0); 
    
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
"""

stan_file = f"{STAN_PATH}/sr_invgamma.stan"
with open(stan_file, "w") as f:
    print(glm_invgamma, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 8: T-Student
glm_tstudent = """
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
    real mu_alpha_0;
    real<lower=0> sigma_alpha_0;
    real<lower=0> student_df;
    real<lower=0> student_scale;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta;
    real<lower=0> tau;
    real mu_chi;
    real<lower=0> sigma_chi;
}

transformed parameters {
    vector[N] mu;
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
    mu_chi ~ normal(mu_alpha_0, sigma_alpha_0);
    sigma_chi ~ normal(0, sigma_alpha_0); 
    
    for (k in 1:p) {
        beta[k] ~ normal(0.0, tau);
    }

    tau ~ student_t(student_df, 0, student_scale); 
    
    for (i in 1:I) {    
        for (j in 1:J) {
            beta_random[i,j] ~ normal(0, sigma_beta);
        }
    }
    sigma_beta ~ student_t(student_df, 0, student_scale);  
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
"""

stan_file = f"{STAN_PATH}/sr_tstudent.stan"
with open(stan_file, "w") as f:
    print(glm_tstudent, file=f)
glm = CmdStanModel(stan_file=stan_file)

# MODEL 9: Half-Normal
glm_normal = """
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
    real mu_alpha_0;
    real<lower=0> sigma_alpha_0;
    real<lower=0> sigma_b;
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta;
    real<lower=0> tau;
    real mu_chi;
    real<lower=0> sigma_chi;
}

transformed parameters {
    vector[N] mu;
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
"""

stan_file = f"{STAN_PATH}/sr_halfnormal.stan"
with open(stan_file, "w") as f:
    print(glm_normal, file=f)
glm = CmdStanModel(stan_file=stan_file)

print(f"\n✓ All Stan models saved to {STAN_PATH}")
print(f"  - baseline_simple_poi.stan")
print(f"  - regularized_horseshoe.stan")
print(f"  - horseshoe.stan")
print(f"  - r2d2.stan")
print(f"  - lasso.stan")
print(f"  - spike_slab.stan")
print(f"  - invgamma.stan")
print(f"  - tstudent.stan")
print(f"  - halfnormal.stan")