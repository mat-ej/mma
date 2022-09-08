import pandas as pd
from util.paths import *

# %%
def load_data(file_name:str = 'per_min_opp_cleaned.csv') -> pd.DataFrame:
    df = pd.read_csv(data / file_name)
    return df
