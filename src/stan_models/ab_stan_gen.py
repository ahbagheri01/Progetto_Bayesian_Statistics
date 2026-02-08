import os

# Import functions
from cmdstanpy import CmdStanModel
from tensorflow_probability.substrates import numpy as tfp
tfd = tfp.distributions

import os
from pathlib import Path

# Find and navigate to Progetto_Bayesian_Statistics
current = Path.cwd()

# Check if we're already there
if current.name == 'Progetto_Bayesian_Statistics':
    print(f"Already in project root: {current}")
else:
    # Search up to parent directories
    for parent in [current] + list(current.parents):
        if parent.name == 'Progetto_Bayesian_Statistics':
            os.chdir(parent)
            print(f"Changed to: {parent}")
            break
    else:
        raise FileNotFoundError("Progetto_Bayesian_Statistics folder not found")

# Verify
assert Path('src').exists(), "src/ folder not found"
print(f"Current directory: {os.getcwd()}")
print(f"Contents: {os.listdir()}")

# Create ./stan folder if does not exists
STAN_PATH = "./src/stan_models/"
if not os.path.exists(STAN_PATH):
    os.mkdir(STAN_PATH)
    
import cmdstanpy
print(cmdstanpy.cmdstan_path())

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
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    // Horseshoe parameters
    vector<lower=0>[p] lambda;
    real<lower=0> tau;
    real<lower=0> c_sq;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    vector<lower=0>[p] lambda_tilde;
    
    // Regularized horseshoe shrinkage
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
    // Likelihood
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
    // Donor priors
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(4, 1);
    sigma_chi ~ normal(0, 1); 
    
    // Regularized horseshoe prior
    lambda ~ cauchy(0, 1);
    tau ~ cauchy(0, p0 / (p - p0) * sigma_chi / sqrt(N));
    c_sq ~ inv_gamma(0.5, 0.5);
    
    for (k in 1:p) {
        beta[k] ~ normal(0, tau * lambda_tilde[k]);
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
"""

stan_file = f"{STAN_PATH}/ab_regularized_horseshoe.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_reg, file=f)
glm = CmdStanModel(stan_file=stan_file)
stan_file = f"{STAN_PATH}/ab_regularized_horseshoe1.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_reg, file=f)
glm = CmdStanModel(stan_file=stan_file)
stan_file = f"{STAN_PATH}/ab_regularized_horseshoe2.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_reg, file=f)
glm = CmdStanModel(stan_file=stan_file)
stan_file = f"{STAN_PATH}/ab_regularized_horseshoe3.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe_reg, file=f)
    
glm = CmdStanModel(stan_file=stan_file)

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
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    // Horseshoe parameters
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
    // Likelihood
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
    // Donor priors
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(4, 1);
    sigma_chi ~ normal(0, 1); 
    
    // Horseshoe prior
    lambda ~ cauchy(0, 1);
    tau ~ cauchy(0, 1);
    
    for (k in 1:p) {
        beta[k] ~ normal(0, tau * lambda[k]);
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
"""

stan_file = f"{STAN_PATH}/ab_horseshoe.stan"
with open(stan_file, "w") as f:
    print(glm_horseshoe, file=f)
glm = CmdStanModel(stan_file=stan_file)

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
    real<lower=0> cons_D2;  // Dirichlet concentration (default 1.0, smaller = more shrinkage)
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    // R2-D2 parameters
    real<lower=0> sigma;  
    simplex[p] phi;       
    real<lower=0, upper=1> R2;
}

transformed parameters {
    vector[N] mu;
    real<lower=0> sigma_beta = sqrt(sigma_beta_sq);
    real<lower=0> omega;  // R²/(1-R²) term
    vector<lower=0>[p] lambda;  // Local shrinkage
    
    // R2-D2 shrinkage: λ_j = sqrt(φ_j) * ω
    omega = sqrt(R2 / (1 - R2));
    for (k in 1:p) {
        lambda[k] = sqrt(phi[k]) * omega;
    }
    
    // Linear predictor
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
    
    // R2-D2 prior (corrected with global scale σ)
    sigma ~ student_t(3, 0, 2.5);  // Global scale prior
    R2 ~ beta(R2_mean * R2_prec, (1 - R2_mean) * R2_prec);
    phi ~ dirichlet(rep_vector(cons_D2, p));
    
    // Regression coefficients: β_j ~ N(0, σ² λ_j²)
    for (k in 1:p) {
        beta[k] ~ normal(0, sigma * lambda[k]);
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
"""

stan_file = f"{STAN_PATH}/ab_r2d2.stan"
with open(stan_file, "w") as f:
    print(glm_r2d2, file=f)
glm = CmdStanModel(stan_file=stan_file)


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
}

parameters {
    vector[p] beta;
    matrix[I, J] beta_random;
    vector<lower=0>[I] alpha;
    real<lower=0> sigma_beta_sq;
    real mu_chi;
    real<lower=0> sigma_chi;
    
    // LASSO parameter
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
    // Likelihood
    Y ~ poisson(rho_trasc .* mu);
    D ~ poisson(rho_donor .* alpha[idx_donor_experiment]);
    
    // Donor priors
    log(alpha) ~ normal(mu_chi, sigma_chi);
    mu_chi ~ normal(4, 1);
    sigma_chi ~ normal(0, 1); 
    
    // LASSO prior
    lambda_lasso ~ cauchy(0, 1);
    beta ~ double_exponential(0, 1.0 / lambda_lasso);
    
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
"""

stan_file = f"{STAN_PATH}/ab_lasso.stan"
with open(stan_file, "w") as f:
    print(glm_lasso, file=f)
glm = CmdStanModel(stan_file=stan_file)

glm_invgamma_54 = """
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

    tau_sq ~ inv_gamma(5, 4); 
    
    for (i in 1:I) {    
        for (j in 1:J) {
            beta_random[i,j] ~ normal(0, sigma_beta);
        }
    }
    sigma_beta_sq ~ inv_gamma(5, 4);  
}

generated quantities {
  vector[N] log_lik;
  for (j in 1:N) {
    log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j]* mu[j]);
  }
}
"""

stan_file = f"{STAN_PATH}/prior_invgamma_54.stan"
with open(stan_file, "w") as f:
    print(glm_invgamma_54, file=f)
glm = CmdStanModel(stan_file=stan_file)

glm_tstudent31 = """
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
    mu_chi ~ normal(0,1);
    sigma_chi ~ normal(0,1); 
    

    for (k in 1:p) {
        beta[k] ~ normal(0.0, tau);
        
    }

    tau ~ student_t(3,0,1); 
    
    for (i in 1:I) {    
        for (j in 1:J) {
            beta_random[i,j] ~ normal(0, sigma_beta);
        }
    }
    sigma_beta ~ student_t(3,0,1);  
 
}

generated quantities {
  vector[N] log_lik;
  for (j in 1:N) {
    log_lik[j] = poisson_lpmf(Y[j] | rho_trasc[j]* mu[j]);
  }
}
"""

stan_file = f"{STAN_PATH}/prior_tstudent_31.stan"
with open(stan_file, "w") as f:
    print(glm_tstudent31, file=f)
glm = CmdStanModel(stan_file=stan_file)


print(f"\n✓ All Stan models saved to {STAN_PATH}")
print(f"  - baseline_simple_poi.stan")
print(f"  - regularized_horseshoe.stan")
print(f"  - horseshoe.stan")
print(f"  - r2d2.stan")
print(f"  - lasso.stan")
print(f"  - prior_invgamma_54.stan\n")
print(f"  - prior_tstudent_31.stan\n")