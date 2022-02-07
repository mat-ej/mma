import csv
import math

import numpy as np
import pandas as pd
import os
import datetime
import locale

from src.preprocessing.features_extraction.feature_variables import (  # isort:skip
    copied_stats,
    red_fighter,
    blue_fighter,
    red_stats,
    blue_stats
)

order = ['WINNER', 'NUM_OF_ROUNDS', 'LAST_ROUND', 'LAST_ROUND_TIME',
         'TOTAL_FIGHT_TIME', 'TITLE_BOUT', 'DECISION_MAJORITY', 'DECISION_SPLIT',
         'DECISION_UNANIMOUS', 'KO/TKO', 'SUBMISSION', 'BANTAMWEIGHT',
         'CATCH WEIGHT', 'FEATHERWEIGHT', 'FLYWEIGHT', 'HEAVYWEIGHT',
         'LIGHT HEAVYWEIGHT', 'LIGHTWEIGHT', 'MIDDLEWEIGHT', 'WELTERWEIGHT',
         'WOMEN\'S BANTAMWEIGHT', 'WOMEN\'S FEATHERWEIGHT', 'WOMEN\'S FLYWEIGHT',
         'WOMEN\'S STRAWWEIGHT', 'REFEREE', 'DATE', 'LOCATION', 'R_ODDS',
         'B_ODDS', 'R_NAME', 'R_AGE', 'R_HEIGHT', 'R_WEIGHT', 'R_REACH', 'R_STYLE', 'R_KD',
         'R_SIG_STR', 'R_SIG_STR_ATT', 'R_TOTAL_STR', 'R_TOTAL_STR_ATT', 'R_TD',
         'R_TD_ATT', 'R_SUB_ATT', 'R_REV', 'R_CTRL', 'R_HEAD', 'R_HEAD_ATT',
         'R_BODY', 'R_BODY_ATT', 'R_LEG', 'R_LEG_ATT', 'R_DISTANCE',
         'R_DISTANCE_ATT', 'R_CLINCH', 'R_CLINCH_ATT', 'R_GROUND',
         'R_GROUND_ATT', 'B_NAME', 'B_AGE', 'B_HEIGHT', 'B_WEIGHT', 'B_REACH', 'B_STYLE',
         'B_KD', 'B_SIG_STR', 'B_SIG_STR_ATT', 'B_TOTAL_STR', 'B_TOTAL_STR_ATT',
         'B_TD', 'B_TD_ATT', 'B_SUB_ATT', 'B_REV', 'B_CTRL', 'B_HEAD',
         'B_HEAD_ATT', 'B_BODY', 'B_BODY_ATT', 'B_LEG', 'B_LEG_ATT',
         'B_DISTANCE', 'B_DISTANCE_ATT', 'B_CLINCH', 'B_CLINCH_ATT', 'B_GROUND',
         'B_GROUND_ATT']

current_file = os.path.abspath(os.path.dirname(__file__))
filename_fights = os.path.join(current_file,
                               '../../../data/preprocessed/cleaned_fights_data_no_nans_onehot_encoded.csv')
filename_styles = os.path.join(current_file,
                               '../../../data/preprocessed/fighters_with_style.csv')
filename_save = os.path.join(current_file,
                             '../../../data/preprocessed/master_with_styles.csv')
fights = pd.read_csv(filename_fights, sep=',')
fighters = pd.read_csv(filename_styles, sep=',')
fights['R_STYLE'] = -1
fights['B_STYLE'] = -1
#print(fights[0, 'R_STYLE'])
for index, fight in fights.iterrows():
    print(index)
    red_name = fight['R_NAME']
    blue_name = fight['B_NAME']

    red_fighter = fighters.loc[fighters['fighter_name'] == red_name]
    blue_fighter = fighters.loc[fighters['fighter_name'] == blue_name]

    if not red_fighter['style'].values.size == 0:
        fights.at[index, 'R_STYLE'] = red_fighter['style']
    if not blue_fighter['style'].values.size == 0:
        fights.at[index, 'B_STYLE'] = blue_fighter['style']

#print(fights.columns)
fights = fights.reindex(columns=order)
fights.to_csv(filename_save, index=False)
