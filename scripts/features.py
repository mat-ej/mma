# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---


# + tags=["parameters"]
upstream = None
product = None
target = None
features = None

# + tags=["injected-parameters"]
# Parameters
features = [
    "TITLE_BOUT",
    "BANTAMWEIGHT",
    "CATCH WEIGHT",
    "FEATHERWEIGHT",
    "FLYWEIGHT",
    "HEAVYWEIGHT",
    "LIGHT HEAVYWEIGHT",
    "LIGHTWEIGHT",
    "MIDDLEWEIGHT",
    "WELTERWEIGHT",
    "WOMENS BANTAMWEIGHT",
    "WOMENS FEATHERWEIGHT",
    "WOMENS FLYWEIGHT",
    "WOMENS STRAWEIGHT",
    "R_AGE",
    "R_HEIGHT",
    "R_REACH",
    "R_WIN_PCT",
    "R_WIN_STREAK",
    "R_LOSS_STREAK",
    "R_KD",
    "R_SIG_STR",
    "R_SIG_STR_ATT",
    "R_TOTAL_STR",
    "R_TOTAL_STR_ATT",
    "R_TD",
    "R_TD_ATT",
    "R_SUB_ATT",
    "R_REV",
    "R_CTRL",
    "R_HEAD",
    "R_HEAD_ATT",
    "R_BODY",
    "R_BODY_ATT",
    "R_LEG",
    "R_LEG_ATT",
    "R_DISTANCE",
    "R_DISTANCE_ATT",
    "R_CLINCH",
    "R_CLINCH_ATT",
    "R_GROUND",
    "R_GROUND_ATT",
    "B_AGE",
    "B_HEIGHT",
    "B_REACH",
    "B_WIN_PCT",
    "B_WIN_STREAK",
    "B_LOSS_STREAK",
    "B_KD",
    "B_SIG_STR",
    "B_SIG_STR_ATT",
    "B_TOTAL_STR",
    "B_TOTAL_STR_ATT",
    "B_TD",
    "B_TD_ATT",
    "B_SUB_ATT",
    "B_REV",
    "B_CTRL",
    "B_HEAD",
    "B_HEAD_ATT",
    "B_BODY",
    "B_BODY_ATT",
    "B_LEG",
    "B_LEG_ATT",
    "B_DISTANCE",
    "B_DISTANCE_ATT",
    "B_CLINCH",
    "B_CLINCH_ATT",
    "B_GROUND",
    "B_GROUND_ATT",
    "R_OPP_KD",
    "R_OPP_SIG_STR",
    "R_OPP_SIG_STR_ATT",
    "R_OPP_TOTAL_STR",
    "R_OPP_TOTAL_STR_ATT",
    "R_OPP_TD",
    "R_OPP_TD_ATT",
    "R_OPP_SUB_ATT",
    "R_OPP_REV",
    "R_OPP_CTRL",
    "R_OPP_HEAD",
    "R_OPP_HEAD_ATT",
    "R_OPP_BODY",
    "R_OPP_BODY_ATT",
    "R_OPP_LEG",
    "R_OPP_LEG_ATT",
    "R_OPP_DISTANCE",
    "R_OPP_DISTANCE_ATT",
    "R_OPP_CLINCH",
    "R_OPP_CLINCH_ATT",
    "R_OPP_GROUND",
    "R_OPP_GROUND_ATT",
    "B_OPP_KD",
    "B_OPP_SIG_STR",
    "B_OPP_SIG_STR_ATT",
    "B_OPP_TOTAL_STR",
    "B_OPP_TOTAL_STR_ATT",
    "B_OPP_TD",
    "B_OPP_TD_ATT",
    "B_OPP_SUB_ATT",
    "B_OPP_REV",
    "B_OPP_CTRL",
    "B_OPP_HEAD",
    "B_OPP_HEAD_ATT",
    "B_OPP_BODY",
    "B_OPP_BODY_ATT",
    "B_OPP_LEG",
    "B_OPP_LEG_ATT",
    "B_OPP_DISTANCE",
    "B_OPP_DISTANCE_ATT",
    "B_OPP_CLINCH",
    "B_OPP_CLINCH_ATT",
    "B_OPP_GROUND",
    "B_OPP_GROUND_ATT",
]
target = ["WINNER"]
upstream = {"data-transform": "/home/m/repo/mma/products/data/data.csv"}
product = {
    "data": "/home/m/repo/mma/products/data/features.csv",
    "nb": "/home/m/repo/mma/products/reports/features.ipynb",
}

# -

import pandas as pd
import numpy as np
# read first upstream
#TODO awkward
upstream_list = [path for path in upstream.values()]
df = pd.read_csv(upstream_list.pop(), parse_dates=['DATE'])
df = df.convert_dtypes()

print("BASIC INFO / SANITY CHECKS")
outcomes = ['DECISION', 'KO', 'SUBMISSION']
print(df[outcomes].sum())
outcomes_prior = df[outcomes].sum().values / len(df)
print(outcomes_prior)

print("DEBUT COUNT = %d" % ((df['R_DEBUT'] + df['B_DEBUT']) >= 1).sum())

df = df[features + target]

nan_cols = df.columns[df.isnull().any()].tolist()
if nan_cols:
    print("FEATURE, TARGET COLUMNS WITH NAN VALUES:")
    print(nan_cols)
else:
    print("NO NANS in features + target cols")


print("MARKET")


df.to_csv(product['data'], index=False)
