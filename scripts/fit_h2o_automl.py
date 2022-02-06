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

# + tags=[]
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# + tags=["parameters"]
upstream = None
product = None
max_models = None
models_path = None
random_seed = None
# + tags=["injected-parameters"]
# This cell was injected automatically based on your stated upstream dependencies (cell above) and pipeline.yaml preferences. It is temporary and will be removed when you save this notebook
max_models = 2
models_path = "/home/m/repo/mma/products/models/h2o/"
random_seed = 1
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    }
}
product = {"nb": "/home/m/repo/mma/products/reports/fit_h2o_automl.ipynb"}


# + tags=[]
target_win = ['WINNER']
names = ['R_NAME', 'B_NAME']
dates = ['DATE']
target_win_odds = ['R_ODDS', 'B_ODDS']
target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']
debuts = ['R_DEBUT', 'B_DEBUT']

# + tags=[]
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
    'TITLE_BOUT',
    'R_DEBUT',
    'B_DEBUT'
    ]


# + tags=[]
train_df = pd.read_csv(upstream['split-train-test']['train'], parse_dates=['DATE'])
test_df = pd.read_csv(upstream['split-train-test']['test'], parse_dates=['DATE'])

# + tags=[]
print(train_df.columns)

# + tags=[]
X_columns = train_df.columns.difference(target_win + target_win_odds + target + target_odds + dates + names).to_list()

# + tags=[]
train_df = train_df[X_columns + target_win].fillna(0.0)
test_df = test_df[X_columns + target_win].fillna(0.0)

# + tags=[]



# + tags=[]
# Start the H2O cluster (locally)
h2o.init()
print(h2o.estimators.xgboost.H2OXGBoostEstimator.available())
train_hf = h2o.H2OFrame(train_df)
test_hf = h2o.H2OFrame(test_df)

# + tags=[]
train_hf[factors] = train_hf[factors].asfactor()
test_hf[factors] = test_hf[factors].asfactor()


# + tags=[]
print(train_hf.describe())

# + tags=[]
aml = H2OAutoML(max_models=10, seed=random_seed, preprocessing = ["target_encoding"], verbosity = 'warn', sort_metric = 'logloss')
aml.train(x=X_columns, y=target_win[0], training_frame=train_hf)
# View the AutoML Leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
print(lb.head())  # Print all rows instead of default (10 rows)
# The leader model is stored here
# print(aml.leader)
# -

print(aml.leader)

# + tags=[]
perf_train = aml.leader.model_performance(train_hf)
print("train accuracy")
perf_train.accuracy()

# + tags=[]
perf_test = aml.leader.model_performance(test_hf)
print("train accuracy")
perf_test.accuracy()

# + tags=[]
exa = aml.leader.explain(train_hf)


# + tags=[]
aml.leader

# + tags=[]
model_path = h2o.save_model(aml.leader, path = models_path, force = True)
my_local_model = h2o.download_model(aml.leader, path=model_path)

# + tags=[]

# # # Get model ids for all models in the AutoML Leaderboard
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# # Get the "All Models" Stacked Ensemble model
# se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in mid][0])
# # Get the Stacked Ensemble metalearner model
# se

# + tags=[]
aml.leader.explain_row(test_hf, row_index=0)

# + tags=[]
# metalearner = se.metalearner()
# metalearner

# + tags=[]
# import os
# import re
# import shutil

# rootdir = models_path
# regex = re.compile('.*AutoML.*')

# for root, dirs, files in os.walk(rootdir):
#   for file in files:
#     if regex.match(file):

#         os.rename(models_path + file, models_path + "automl")

# + tags=[]
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# GLM = h2o.get_model([mid for mid in model_ids if "GLM" in mid][0])
# XGBOOST = h2o.get_model([mid for mid in model_ids if "XGB" in mid][0])
# ENSEMBLE = h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in mid][0])
# models_to_save = [(GLM, "glm"), (XGBOOST, "xgboost"), (ENSEMBLE, "ensemble")]
