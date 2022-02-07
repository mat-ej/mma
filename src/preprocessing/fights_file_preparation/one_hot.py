import csv
import math

import numpy as np
import pandas as pd
import os
import datetime
import locale

from sklearn.preprocessing import OneHotEncoder

current_file = os.path.abspath(os.path.dirname(__file__))
filename_fights = os.path.join(current_file, '../../../data/preprocessed/cleaned_fights_data.csv')
fights_df = pd.read_csv(filename_fights, sep=',')
enc = OneHotEncoder(handle_unknown='ignore', dtype=int)


win_by = pd.DataFrame(enc.fit_transform(fights_df[['WIN_BY']]).toarray())
column_names = ['DECISION_MAJORITY', 'DECISION_SPLIT', 'DECISION_UNANIMOUS', 'KO/TKO', 'SUBMISSION']
win_by.columns = column_names
fights_df[column_names] = win_by
fights_df.drop(columns=['WIN_BY'], inplace=True)
print(fights_df.columns)

weight_category = pd.DataFrame(enc.fit_transform(fights_df[['WEIGHT_CATEGORY']]).toarray())
column_names = [x.upper() for x in enc.categories_[0]]
fights_df[column_names] = weight_category
fights_df.drop(columns=['WEIGHT_CATEGORY'], inplace=True)



target_stats = [
    'WINNER',
    'NUM_OF_ROUNDS',
    'LAST_ROUND',
    'LAST_ROUND_TIME',
    'TOTAL_FIGHT_TIME',
    'TITLE_BOUT',
'DECISION_MAJORITY', 'DECISION_SPLIT',
       'DECISION_UNANIMOUS', 'KO/TKO', 'SUBMISSION', 'BANTAMWEIGHT',
       'CATCH WEIGHT', 'FEATHERWEIGHT', 'FLYWEIGHT', 'HEAVYWEIGHT',
       'LIGHT HEAVYWEIGHT', 'LIGHTWEIGHT', 'MIDDLEWEIGHT', 'WELTERWEIGHT',
       'WOMEN\'S BANTAMWEIGHT', 'WOMEN\'S FEATHERWEIGHT', 'WOMEN\'S FLYWEIGHT',
       'WOMEN\'S STRAWWEIGHT',
    'REFEREE',
    'DATE',
    'LOCATION',
    'R_ODDS',
    'B_ODDS',
    'R_NAME',
    'R_AGE',
    'R_HEIGHT',
    'R_WEIGHT',
    'R_REACH',
    'R_KD',
    'R_SIG_STR',
    'R_SIG_STR_ATT',
    'R_TOTAL_STR',
    'R_TOTAL_STR_ATT',
    'R_TD',
    'R_TD_ATT',
    'R_SUB_ATT',
    'R_REV',
    'R_CTRL',
    'R_HEAD',
    'R_HEAD_ATT',
    'R_BODY',
    'R_BODY_ATT',
    'R_LEG',
    'R_LEG_ATT',
    'R_DISTANCE',
    'R_DISTANCE_ATT',
    'R_CLINCH',
    'R_CLINCH_ATT',
    'R_GROUND',
    'R_GROUND_ATT',
    'B_NAME',
    'B_AGE',
    'B_HEIGHT',
    'B_WEIGHT',
    'B_REACH',
    'B_KD',
    'B_SIG_STR',
    'B_SIG_STR_ATT',
    'B_TOTAL_STR',
    'B_TOTAL_STR_ATT',
    'B_TD',
    'B_TD_ATT',
    'B_SUB_ATT',
    'B_REV',
    'B_CTRL',
    'B_HEAD',
    'B_HEAD_ATT',
    'B_BODY',
    'B_BODY_ATT',
    'B_LEG',
    'B_LEG_ATT',
    'B_DISTANCE',
    'B_DISTANCE_ATT',
    'B_CLINCH',
    'B_CLINCH_ATT',
    'B_GROUND',
    'B_GROUND_ATT'
]

order = ['DECISION_MAJORITY', 'DECISION_SPLIT',
       'DECISION_UNANIMOUS', 'KO/TKO', 'SUBMISSION', 'BANTAMWEIGHT',
       'CATCH WEIGHT', 'FEATHERWEIGHT', 'FLYWEIGHT', 'HEAVYWEIGHT',
       'LIGHT HEAVYWEIGHT', 'LIGHTWEIGHT', 'MIDDLEWEIGHT', 'WELTERWEIGHT',
       'WOMEN\'S BANTAMWEIGHT', 'WOMEN\'S FEATHERWEIGHT', 'WOMEN\'S FLYWEIGHT',
       'WOMEN\'S STRAWWEIGHT']



# fights_df = fights_df.reindex(columns=target_stats)
print(fights_df.columns)
print(fights_df.head())

# fights_df.drop(columns=['WIN_BY'], inplace=True)
# fights_df.drop(columns=['WEIGHT_CATEGORY'], inplace=True)

filename = os.path.join(current_file, '../../../data/preprocessed/cleaned_fights_data_one_hot.csv')
fights_df.to_csv(filename, index=False)


#print(enc.categories)

