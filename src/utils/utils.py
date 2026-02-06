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

def transform_data():
    #loading data
    #get the data
    data=pd.read_csv("./data/data.csv")
    data_donors=pd.read_csv("./data/data_donatori.csv")
    data=data[data['Conta']!='TMTC']
    data_donors=data_donors[data_donors['Conta']!='TMTC']
    #Rename columns
    data.rename(columns={'Idx Replica Condizione': 'Idx Replica Experiment','Controllo':'Control','Idx Replica Diluizione':'Idx Replica Diluition','Conta':'Count'}, inplace=True)
    data_donors.rename(columns={'Idx Replica Diluizione':'Idx Replica Experiment','Conta':'Count'}, inplace=True)

    #Modify values in some columns:
    #Modify Idx Experiment such that experiments 5 and 6 have the same idx, experiments 7 and 8, experiment 5,6 has the same index as 5 and 6 and experiment 7,8 the same as 7 and 8
    #but first convert all to string to avoid problems with mixed types
    data['Idx Experiment'] = data['Idx Experiment'].astype(str)
    data['Idx Experiment'] = data['Idx Experiment'].replace({'6': '5', '7':'6','8': '6', '9':'7','10':'8','5,6': '5', '7,8':'6'})
    #now reconvert to numeric
    data['Idx Experiment'] = pd.to_numeric(data['Idx Experiment'])

    data_donors['Idx Experiment'] = data_donors['Idx Experiment'].astype(str)
    data_donors['Idx Experiment'] = data_donors['Idx Experiment'].replace({'6': '5', '7':'6','8': '6', '9':'7','10':'8','5,6': '5', '7,8':'6'})

    #modify control column: convert 'Yes' to 1 and 'No' to 0
    data['Control'] = data['Control'].replace({'Yes': 1, 'No': 0})



    #now reconvert to numeric
    data_donors['Idx Experiment'] = pd.to_numeric(data_donors['Idx Experiment'])

    print(data['Idx Experiment'].unique())
    print(data_donors['Idx Experiment'].unique())

    #convert to numeric all the others columns I need 
    #convert A;B;C in 1;2;3
    data['Idx Replica Experiment'] = data['Idx Replica Experiment'].astype(str)
    data['Idx Replica Experiment'] = data['Idx Replica Experiment'].replace({'A': '1', 'B':'2','C': '3'})
    #now reconvert to numeric
    data['Idx Replica Experiment'] = pd.to_numeric(data['Idx Replica Experiment'])

    data['Count'] = pd.to_numeric(data['Count'])
    data['Idx Replica Diluition'] = pd.to_numeric(data['Idx Replica Diluition'])
    data['Diluition'] = pd.to_numeric(data['Diluition'])
    data['IBU'] = pd.to_numeric(data['IBU'])
    data['DMSO'] = pd.to_numeric(data['DMSO'])
    data['Temp'] = pd.to_numeric(data['Temp'])
    data_donors['Count'] = pd.to_numeric(data_donors['Count'])
    data_donors['Idx Replica Experiment'] = pd.to_numeric(data_donors['Idx Replica Experiment'])  
    return data, data_donors





def get_glm_data(data, data_donors, interactions_normalizer = lambda x : x, quadratic_normalizer = lambda x : x - np.mean(x)):
    #Extract the data we need for the Stan model 
#from trasconjugant dataset
    Y=data.Count.values
    N=len(Y)
    idx_experiment=data['Idx Experiment'].values
    I=len(np.unique(idx_experiment))
    idx_experiment_replica=data['Idx Replica Experiment'].values
    J=len(np.unique(idx_experiment_replica))
    dil_trasc=data['Diluition'].values
    rho_trasc = np.power(10.0, dil_trasc)


#from donor dataset
    D=data_donors.Count.values
    M=len(D)
    idx_donor_experiment=data_donors['Idx Experiment'].values
    dil_donor=data_donors['Diluition'].values
    rho_donor=np.power(10.0, dil_donor)


    #build design matrix X
    IBU=data['IBU'].values
    DMSO=data['DMSO'].values
    Temp=data['Temp'].values
    
    normalizer = lambda x : (x - np.mean(x)) / np.std(x)
    #normalize covariates
    IBU = normalizer(IBU)  
    DMSO = normalizer(DMSO)
    Temp = normalizer(Temp)
    IBUDMSO = IBU * DMSO   
    IBUTemp = IBU * Temp
    DMSOTemp = DMSO * Temp
    IBU2 = IBU**2  
    DMSO2 = DMSO**2
    Temp2 = Temp**2
    

    X = np.column_stack([np.ones_like(Y),IBU,DMSO,Temp])
    X_WITH_INTERACTIONS = np.column_stack([
        np.ones_like(Y),     # intercept
        IBU,DMSO,Temp,
        interactions_normalizer(IBUDMSO),          # interactions
        interactions_normalizer(IBUTemp),
        interactions_normalizer(DMSOTemp)
    ])
    
    X_FULL_Q = np.column_stack([
        np.ones_like(Y),     # intercept
        IBU,                  # main effects
        DMSO,
        Temp,
        interactions_normalizer(IBU * DMSO),          # interactions with mean reduced.
        interactions_normalizer(IBU * Temp),
        interactions_normalizer(DMSO * Temp),
        quadratic_normalizer(IBU2),
        quadratic_normalizer(DMSO2),
        quadratic_normalizer(Temp2)    
    ])
    
    glm_data = {
    "N": N,
    "M": M,
    "I": I,
    "J": J,
    "Y": Y,
    "D": D,
    "rho_trasc": rho_trasc,
    "rho_donor": rho_donor,
    "idx_experiment": idx_experiment,
    "idx_experiment_replica": idx_experiment_replica,
    "idx_donor_experiment": idx_donor_experiment}
    return glm_data, {"X": X, "X2_I": X_WITH_INTERACTIONS, "X2": X_FULL_Q}

