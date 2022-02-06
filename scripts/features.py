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
filename = '../data/per_min_debuts.csv'

# + tags=["injected-parameters"]
# Parameters
filename = "data/per_min_debuts.csv"
product = {"data": "/home/m/repo/mma/products/data/features.csv", "nb": "/home/m/repo/mma/products/nb/features.ipynb"}

# -

import pandas as pd

df = pd.read_csv(filename, parse_dates=['DATE'])

drop = ['R_WEIGHT', 'B_WEIGHT']
market = ['R_ODDS', 'B_ODDS', 'R_DEC_ODDS', 'B_DEC_ODDS', 'R_SUB_ODDS',
       'B_SUB_ODDS', 'R_KO_ODDS', 'B_KO_ODDS']

df.drop(columns=drop)

df.rename(columns={'KO/TKO':'KO'}, inplace=True)
df['DECISION'] = ((df.DECISION_SPLIT + df.DECISION_MAJORITY + df.DECISION_UNANIMOUS) > 0).astype(int)
df.drop(columns=['DECISION_MAJORITY', 'DECISION_SPLIT', 'DECISION_UNANIMOUS'], inplace=True)
df = df.convert_dtypes()

outcomes = ['DECISION', 'KO', 'SUBMISSION']
print(df[outcomes].sum())
outcomes_prior = df[outcomes].sum().values / len(df)
print(outcomes_prior)

target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']

df['R_DEC'] = df.WINNER.astype(bool) * df.DECISION.astype(bool)
df['R_KO'] = df.WINNER.astype(bool) * df.KO.astype(bool)
df['R_SUB'] = df.WINNER.astype(bool) * df.SUBMISSION.astype(bool)

df['B_DEC'] = ~df.WINNER.astype(bool) * df.DECISION.astype(bool)
df['B_KO'] = ~df.WINNER.astype(bool) * df.KO.astype(bool)
df['B_SUB'] = ~df.WINNER.astype(bool) * df.SUBMISSION.astype(bool)
df[target] = df[target].astype(int)

df.drop(columns=['DECISION', 'KO', 'SUBMISSION', 'R_WEIGHT', 'B_WEIGHT'], inplace=True)

df.to_csv(product['data'], index=False)
