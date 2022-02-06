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
# + tags=["injected-parameters"]
# Parameters
upstream = {
    "autosklearn": {
        "nb": "/home/m/repo/mma/products/reports/fit-sklearn-automl.ipynb",
        "model": "/home/m/repo/mma/products/models/sklearn-automl.pickle",
    },
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    },
}
product = {"nb": "/home/m/repo/mma/products/reports/final_evaluation.ipynb"}

# -

target_win = ['WINNER']
names = ['R_NAME', 'B_NAME']
dates = ['DATE']
target_win_odds = ['R_ODDS', 'B_ODDS']
target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']


import pandas as pd
import numpy as np
import pickle
test_df = pd.read_csv(upstream['split-train-test']['test'])
model = pickle.load(open(upstream['autosklearn']['model'], 'rb'))

print(model)
print(test_df)

X_columns = test_df.columns.difference(target_win + target_win_odds + target + target_odds + names + dates).to_list()

X_test = test_df[X_columns]
Y_test = test_df['WINNER'].values
Y_model_probs = model.predict_proba(X_test)
Y_model = model.predict(X_test)
model_acc = np.sum(Y_test == Y_model) / len(Y_test)
print(model_acc)

odds = test_df[target_win_odds]
Y_market = (odds.R_ODDS < odds.B_ODDS).values
market_acc = np.sum(Y_test == Y_market) / len(Y_test)

print(market_acc)






