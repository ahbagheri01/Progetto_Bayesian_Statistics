# Bayesian Modeling of Transconjugation Efficacy Under Chemical Stressors

Hierarchical Bayesian GLM (Poisson & Negative Binomial) to estimate transconjugation rates in *E. coli* under temperature, ibuprofen, and DMSO stressors. Implemented in Stan via CmdStanPy.

## Setup

```bash
git clone https://github.com/ahbagheri01/Progetto_Bayesian_Statistics
cd Progetto_Bayesian_Statistics

# Create required directories
mkdir -p data logs results

# Place transconjugant and donor count datasets into data/
```

## Project Structure

```
├── data/                   # Raw datasets (transconjugants + donors)
├── logs/                   # MCMC sampling logs
├── results/                # Posterior outputs, plots, 
└── src/
    ├── modules/
    │   └── sampler.py      # MCMC sampler configuration
    └── stan_models/
        ├── ab_stan_gen/    # Poisson models
        └── ab_stan_neg/    # Negative Binomial models
```

## Usage

### Configure the sampler

Edit `src/modules/sampler.py` to adjust chains, iterations, warmup, etc., then run:

```bash
python3 -m src.modules.sampler
```

### Run a model

```bash
# Negative Binomial model — N is number of parallel threads
python3 -m src.stan_models.ab_stan_neg [N]

# Poisson model
python3 -m src.stan_models.ab_stan_gen [N]
```

### Add a custom model

Create a new Stan file under `src/stan_models/` and wire it to a Python runner following the pattern in `ab_stan_neg`.

## Models

| Likelihood | Design Matrix | Shrinkage Prior | Selected Covariates |
|---|---|---|---|
| Poisson | X | Simple, Lasso, R2D2, Spike&Slab | DMSO |
| Poisson | X_interactions | Simple, Lasso, Horseshoe, R2D2, Spike&Slab | DMSO, IBU×Temp |
| Negative Binomial | X, X_interactions | All | — (overdispersion absorbs signal) |

> **Note:** NB selecting no covariates is expected behavior — the overdispersion parameter absorbs the variance that Poisson attributes to covariates. Not a bug.

## Key Results

- **DMSO** has a consistent negative effect on transconjugation rate across all Poisson models
- **IBU** has no clear main effect but interacts significantly with temperature
- **NB outperforms Poisson** in posterior predictive checks: 3 vs 87 outliers out of 378 observations
- Results are **robust to prior specification** across 20 sensitivity analysis models (LOO variance < 8 points)
- Models with **>5% divergence rate are excluded** from inference
- Quadratic term models dropped due to posterior instability

## Computational Notes

- Total compute: ~77.5 hours across 128 configurations
- Hardware: Intel Core i7-12650H, 8 GB RAM, Ubuntu 24.04
- Sampler: Stan via CmdStanPy