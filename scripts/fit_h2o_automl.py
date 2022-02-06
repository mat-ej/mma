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

import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# + tags=["parameters"]
upstream = None
product = None
max_models = None
# + tags=["injected-parameters"]
# Parameters
max_models = 20
upstream = {"split-train-test": {"train": "/home/m/repo/mma/products/data/train.csv", "test": "/home/m/repo/mma/products/data/test.csv"}}
product = {"nb": "/home/m/repo/mma/products/reports/fit_h2o_automl.ipynb", "model": "/home/m/repo/mma/products/models/h2o_automl.pickle"}
# -

target_win = ['WINNER']
names = ['R_NAME', 'B_NAME']
dates = ['DATE']
target_win_odds = ['R_ODDS', 'B_ODDS']
target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']
debuts = ['R_DEBUT', 'B_DEBUT']

factors = ['TITLE_BOUT',
    'BANTAMWEIGHT',
    'CATCH WEIGHT',
    'FEATHERWEIGHT',
    'FLYWEIGHT',
    'HEAVYWEIGHT',
    'LIGHT HEAVYWEIGHT',
    'LIGHTWEIGHT',
    'MIDDLEWEIGHT',
    'WELTERWEIGHT',
    'WOMEN\'S BANTAMWEIGHT',
    'WOMEN\'S FEATHERWEIGHT',
    'WOMEN\'S FLYWEIGHT',
    'WOMEN\'S STRAWWEIGHT',
    'WINNER',
    'TITLE_BOUT'
]


# df_clean = df[X_columns + target_win].dropna()
# train_df, test_df = split_train_test(df_clean, 0.3)

train_df = pd.read_csv(upstream['split-train-test']['train'], parse_dates=['DATE'])
test_df = pd.read_csv(upstream['split-train-test']['test'], parse_dates=['DATE'])

print(train_df.columns)

X_columns = train_df.columns.difference(target_win + target_win_odds + target + target_odds + dates + names + debuts).to_list()

train_df = train_df[X_columns + target_win].dropna()
test_df = test_df[X_columns + target_win].dropna()



# Start the H2O cluster (locally)
h2o.init()
print(h2o.estimators.xgboost.H2OXGBoostEstimator.available())
train_hf = h2o.H2OFrame(train_df)
test_hf = h2o.H2OFrame(test_df)

train_hf[factors] = train_hf[factors].asfactor()
test_hf[factors] = test_hf[factors].asfactor()


print(train_hf.describe())

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=max_models, seed=1)
aml.train(x=X_columns, y=target_win[0], training_frame=train_hf)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)
print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)
# The leader model is stored here
print(aml.leader)

h2o.save_model(aml.leader, path = product['model'])
# # Get model ids for all models in the AutoML Leaderboard
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# # Get the "All Models" Stacked Ensemble model
# se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# # Get the Stacked Ensemble metalearner model
# metalearner = se.metalearner()
#



