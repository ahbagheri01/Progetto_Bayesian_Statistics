import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import os
import arviz as az

# Import functions
from cmdstanpy import CmdStanModel
from tensorflow_probability.substrates import numpy as tfp
tfd = tfp.distributions


import cmdstanpy
import os
from pathlib import Path

import os, sys
from pathlib import Path
def set_project_root():
        
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
    print(cmdstanpy.cmdstan_path())
    return STAN_PATH

STAN_PATH = set_project_root()
print("cwd:", os.getcwd())
print("first sys.path entries:")
for p in sys.path[:5]:
    print("  ", p)

print("project root guess:", Path().resolve())
print("exists ./src ?", (Path().resolve() / "src").exists())

from src.utils.utils import *


data, data_donors = transform_data()
glm_data, X_D = get_glm_data(data, data_donors)


model_names = [
    ("ab_baseline_simple_poi.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "tau_sq_a": 2.0,
        "tau_sq_b": 1.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_lasso.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_horseshoe.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_regularized_horseshoe.stan", {
        "p0": 2,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_r2d2.stan", {
        "R2_mean": 0.5,
        "R2_prec": 2.0,
        "cons_D2": 0.5,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }),
    ("ab_spike_slab.stan", {
        "pi_prior": 0.2,
        "tau_slab": 2.0,
        "tau_spike": 0.01,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }),
     ("ab_neg_binomial.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "tau_sq_a": 2.0,
        "tau_sq_b": 1.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,

        # NEW: dispersion prior rates for Pascal/NB2
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1,
    })
    # ("sr_invgamma.stan", {
    #     "mu_alpha_0": 0.0,
    #     "sigma_alpha_0": 10.0,
    #     "tau_sq_a": 2.0,
    #     "tau_sq_b": 1.0,
    #     "sigma_beta_sq_a": 2.0,
    #     "sigma_beta_sq_b": 1.0
    # }),
    # ("sr_tstudent.stan", {
    #     "mu_alpha_0": 0.0,
    #     "sigma_alpha_0": 10.0,
    #     "student_df": 3.0,
    #     "student_scale": 2.5
    # }),
    # ("sr_halfnormal.stan", {
    #     "mu_alpha_0": 0.0,
    #     "sigma_alpha_0": 10.0,
    #     "sigma_b": 2.5
    # })
]


model_names = [
    # POISSON MODELS
    ("ab_baseline_simple_poi.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "tau_sq_a": 2.0,
        "tau_sq_b": 1.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_lasso.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_horseshoe.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_regularized_horseshoe.stan", {
        "p0": 2,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }), 
    ("ab_r2d2.stan", {
        "R2_mean": 0.5,
        "R2_prec": 2.0,
        "cons_D2": 0.5,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }),
    ("ab_spike_slab.stan", {
        "pi_prior": 0.2,
        "tau_slab": 2.0,
        "tau_spike": 0.01,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0
    }),
    
    # NEGATIVE BINOMIAL MODELS
    ("ab_baseline_negbin.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "tau_sq_a": 2.0,
        "tau_sq_b": 1.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    }),
    ("ab_lasso_negbin.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    }),
    ("ab_horseshoe_negbin.stan", {
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    }),
    ("ab_regularized_horseshoe_negbin.stan", {
        "p0": 2,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    }),
    ("ab_r2d2_negbin.stan", {
        "R2_mean": 0.5,
        "R2_prec": 2.0,
        "cons_D2": 0.5,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    }),
    ("ab_spike_slab_negbin.stan", {
        "pi_prior": 0.2,
        "tau_slab": 2.0,
        "tau_spike": 0.01,
        "mu_chi_mu": 0.0,
        "mu_chi_sigma": 10.0,
        "sigma_chi_sigma": 5.0,
        "sigma_beta_sq_a": 2.0,
        "sigma_beta_sq_b": 1.0,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    }),
    ("sr_halfnormal_negbin.stan", {
        "mu_alpha_0": 0.0,
        "sigma_alpha_0": 10.0,
        "sigma_b": 2.5,
        "phi_y_rate": 0.1,
        "phi_d_rate": 0.1
    })
]

models = {}
for key in model_names:
    print(f"Compiling model: {key[0]}")
    glm = CmdStanModel(stan_file=f"{STAN_PATH}/{key[0]}")
    for x in X_D:
        if x == "X2":
            continue
        models[(key[0], tuple(key[1].items()), x)] = {"model": glm, "X": X_D[x]}
        new_data = glm_data.copy()
        new_data.update({"X": X_D[x], "p": X_D[x].shape[1]})
        new_data.update(key[1])
        models[(key[0], tuple(key[1].items()), x)]["data"] = new_data

os.makedirs("./results/ab_hyp/", exist_ok=True)

# CHECK WHAT EXISTS VS WHAT NEEDS SAMPLING
existing = []
to_sample = []
for key in models.keys():
    filename = f"./results/ab_hyp/{'_'.join(str(k) for k in key)}.nc"
    if os.path.exists(filename):
        existing.append(key)
    else:
        to_sample.append(key)

print(f"\n{'='*70}")
print(f"ALREADY EXISTS: {len(existing)}")
for k in existing:
    print(f"  ✓ {k}")
print(f"\nWILL SAMPLE: {len(to_sample)}")
for k in to_sample:
    print(f"  → {k}")
print(f"{'='*70}\n")

def sample_model(key, model_config, sample_config):
    glm_fit = model_config["model"].sample(
        data=model_config["data"], 
        **sample_config
    )
    return key, glm_fit

sample_config = {
    "chains": 4,  
    "parallel_chains": 4,
    "iter_warmup": 2000, 
    "iter_sampling": 6000,
    "show_console": False,
    "adapt_delta": 0.99,
    "max_treedepth": 12, 
}

max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
print(f"Using max_workers={max_workers}\n")
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(sample_model, k, models[k], sample_config): k 
        for k in to_sample  # Only submit what needs sampling
    }
    
    for future in as_completed(futures):
        key = futures[future]
        key, fit = future.result()
        models[key]["fit"] = fit
        models[key]["az"] = az.from_cmdstanpy(fit)
        filename = f"./results/ab_hyp/{'_'.join(str(k) for k in key)}.nc"
        az.to_netcdf(models[key]["az"], filename)
        print(f"✓ Saved: {key}")

print("\n✓ Done")
