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

from src.utils.utils import set_project_root, transform_data, get_glm_data

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



model_names = [("ab_baseline_simple_poi.stan", {}), ("ab_lasso.stan", {}), ("ab_horseshoe.stan", {}), ("ab_regularized_horseshoe.stan", {"p0": 2}), ("ab_r2d2.stan", {"R2_mean": 0.5,"R2_prec": 2.0, "cons_D2": 0.5})]

models = {}
for key in model_names:
    print(f"Compiling model: {key}")
    glm = CmdStanModel(stan_file=f"{STAN_PATH}/{key[0]}")
    for x in X_D:
        models[(key[0], tuple(key[1].items()), x)] = {"model": glm, "X": X_D[x]}
        new_data = glm_data.copy()
        new_data.update({"X": X_D[x], "p": X_D[x].shape[1]})
        new_data.update(key[1])
        models[(key[0], tuple(key[1].items()), x)]["data"] = new_data
        
        
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "show_console": True,
    "adapt_delta": 0.99,  # increase from 0.8
    "max_treedepth": 12, 
}


max_workers = 8
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(sample_model, k, v, sample_config): k 
        for k, v in models.items()
    }
    for future in as_completed(futures):
        key, fit = future.result()
        models[key]["fit"] = fit
        models[key]["az"] = az.from_cmdstanpy(fit)
        print(f"✓ Completed: {key}")

print("\n✓ All models fitted")


import os

os.makedirs("./results/ab_results/", exist_ok=True)

for key in models:
    poi_glm_data = models[key]["az"]
    
    # Print divergences
    print(f"Divergences {key}: {np.sum(poi_glm_data.sample_stats.diverging.values)}")
    
    # Create filename from key tuple
    filename = f"{'_'.join(str(k) for k in key)}.nc"
    az.to_netcdf(poi_glm_data, f"./results/ab/{filename}")