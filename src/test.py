# %% [markdown]
# ### Prior elicitation 

# %%
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

# %%
data, data_donors = transform_data()
glm_data, X_D = get_glm_data(data, data_donors)

# %%
model_names = model_names = [
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.18, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.18, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.18, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.18, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.72, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.72, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.72, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.72, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":2.0, "sigma_beta_sq_a":3, "sigma_beta_sq_b":2.0, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":2.0, "sigma_beta_sq_a":3, "sigma_beta_sq_b":2.0, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.08, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.08, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.08, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.08, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":4.5, "sigma_beta_sq_a":3, "sigma_beta_sq_b":4.5, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":4.5, "sigma_beta_sq_a":3, "sigma_beta_sq_b":4.5, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.18, "sigma_beta_sq_a":3, "sigma_beta_sq_b":2.0, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":0.18, "sigma_beta_sq_a":3, "sigma_beta_sq_b":2.0, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":2.0, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.18, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":3, "tau_sq_b":2.0, "sigma_beta_sq_a":3, "sigma_beta_sq_b":0.18, "mu_alpha_0":0, "sigma_alpha_0":10}),


    ("sr_invgamma.stan", {"tau_sq_a":2, "tau_sq_b":0.09, "sigma_beta_sq_a":2, "sigma_beta_sq_b":0.09, "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":2, "tau_sq_b":0.09, "sigma_beta_sq_a":2, "sigma_beta_sq_b":0.09, "mu_alpha_0":0, "sigma_alpha_0":10}),
    ("sr_invgamma.stan", {"tau_sq_a":2, "tau_sq_b":1.0,  "sigma_beta_sq_a":2, "sigma_beta_sq_b":1.0,  "mu_alpha_0":0, "sigma_alpha_0":5}),
    ("sr_invgamma.stan", {"tau_sq_a":2, "tau_sq_b":1.0,  "sigma_beta_sq_a":2, "sigma_beta_sq_b":1.0,  "mu_alpha_0":0, "sigma_alpha_0":10}),
    
    ("sr_tstudent.stan", {"student_df":3,"student_scale":0.5,"mu_alpha_0":0,"sigma_alpha_0":1}),
    ("sr_tstudent.stan", {"student_df":3,"student_scale":0.5,"mu_alpha_0":0,"sigma_alpha_0":10}),
    ("sr_tstudent.stan", {"student_df":3,"student_scale":0.5,"mu_alpha_0":0,"sigma_alpha_0":10}),
    ("sr_halfnormal.stan", {"sigma_b":0.5,"mu_alpha_0":0,"sigma_alpha_0":1}),
    ("sr_halfnormal.stan", {"sigma_b":0.5,"mu_alpha_0":0,"sigma_alpha_0":10}),
    ("sr_halfnormal.stan", {"sigma_b":1,"mu_alpha_0":0,"sigma_alpha_0":10})
]

models = {}
for key in model_names:
    print(f"Compiling model: {key}")
    glm = CmdStanModel(stan_file=f"{STAN_PATH}/{key[0]}")
    for x in X_D:
        if x == "X2":
            print(f"Fitting model: {key} with {x} (skipped)")
            continue
        models[(key[0], tuple(key[1].items()), x)] = {"model": glm, "X": X_D[x]}
        new_data = glm_data.copy()
        new_data.update({"X": X_D[x], "p": X_D[x].shape[1]})
        new_data.update(key[1])
        models[(key[0], tuple(key[1].items()), x)]["data"] = new_data

# %%
len(models)

# %%
import os
remove_keys = []
for key in models:
    filename = f"{'_'.join(str(k) for k in key)}.nc"
    try:
        models[key]["az"] = az.from_netcdf(f"./results/sr_prolic/{filename}")
    except FileNotFoundError:
        remove_keys.append(key)
        pass




# %%
for k in remove_keys:
    print(f"Removing {k} due to missing file")
    del models[k]

# %%
len(models)

# %%
model_names = list(models.keys())
# models[model_names[0]]["fit"].summary()
# %%
compare_dict = {
}
for i,key in enumerate(models):
    compare_dict[i] = models[key]["az"]
    print(f"Added model {i}: {key} to comparison dict")

# %%
comparison_waic = az.compare(compare_dict, ic="waic")
print("\n=== WAIC COMPARISON ===")
print(comparison_waic)

az.plot_compare(comparison_waic, insample_dev=False)
plt.tight_layout()
plt.savefig("results/sr_waic_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("Saved → results/sr_waic_comparison.png")

# %%
comparison_loo = az.compare(compare_dict, ic="loo")
print("\n=== LOO COMPARISON ===")
print(comparison_loo)

az.plot_compare(comparison_loo, insample_dev=False)
plt.savefig("results/sr_loo_comparison.png", dpi=150, bbox_inches='tight')
plt.tight_layout()
plt.show()
    
for i,key in enumerate(models):
    print(f"Added model {i}: {key} to comparison dict")

