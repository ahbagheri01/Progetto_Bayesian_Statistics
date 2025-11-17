import dotenv
import os
import pandas as pd
dotenv.load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
data1 = pd.read_csv(f'{DATA_PATH}/data.csv')
data_eng = data1.rename(columns={
    'Idx Experiment': 'Experiment Index',
    'Idx Replica Diluizione': 'Dilution Replicate Index',
    'Idx Replica Condizione': 'Condition Replicate Index',
    'Controllo': 'Control',
    'Diluition': 'Dilution',
    'IBU': 'IBU',
    'DMSO': 'DMSO',
    'Temp': 'Temperature',
    'Conta': 'Count'
})
data_eng.to_csv(f'{DATA_PATH}/data_eng.csv', index=False)

data2 = pd.read_csv('data_donatori.csv')
data_donatori_eng = data2.rename(columns={
    'Idx Experiment': 'Experiment Index',
    'Idx Replica Diluizione': 'Dilution Replicate Index',
    'Diluition': 'Dilution',
    'IBU': 'IBU',
    'DMSO': 'DMSO',
    'Temp': 'Temperature',
    'Conta': 'Count'
})
data_donatori_eng.to_csv(f'{DATA_PATH}/data_donatori_eng.csv', index=False)

# TODO Merge two datasets into a single file with an additional column indicating the source
