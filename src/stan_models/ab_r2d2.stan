
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

