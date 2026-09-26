import pandas as pd

from src.config import DATASET_PATH

def load_data():
    #load the vehicle fuel consumption dataset
    
    dataframe = pd.read_csv(DATASET_PATH)
    return dataframe
