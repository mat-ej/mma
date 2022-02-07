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
models_path = None
random_seed = None
target = None
factors = None
include_algos = None
# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
max_models = 2
models_path = "/home/m/repo/mma/products/models/h2o/"
random_seed = 1
factors = [
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
]
include_algos = ["GLM", "DRF", "DeepLearning"]
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    }
}
product = {"nb": "/home/m/repo/mma/products/reports/fit_h2o_automl.ipynb"}

# -

import pandas as pd
train_df = pd.read_csv(upstream['split-train-test']['train'])
test_df = pd.read_csv(upstream['split-train-test']['test'])

X_train = train_df.drop(columns = target)
Y_train = train_df[target]

X_test = test_df.drop(columns = target)
Y_test = test_df[target]

print(train_df.columns)

X_columns = X_test.columns

# Start the H2O cluster (locally)
h2o.init()
print(h2o.estimators.xgboost.H2OXGBoostEstimator.available())
train_hf = h2o.H2OFrame(train_df)
test_hf = h2o.H2OFrame(test_df)

train_hf[factors] = train_hf[factors].asfactor()
test_hf[factors] = test_hf[factors].asfactor()

train_hf[target] = train_hf[target].asfactor()
test_hf[target] = test_hf[target].asfactor()


print(train_hf.describe())

aml = H2OAutoML(max_models=max_models,
                seed=random_seed,
                verbosity = 'warn',
                include_algos = include_algos
                )

aml.train(x=X_train.columns.to_list(), y=target[0], training_frame=train_hf)
# View the AutoML Leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
print(lb.head())  # Print all rows instead of default (10 rows)
# The leader model is stored here
print(aml.leader)



perf_train = aml.leader.model_performance(train_hf)
print("train accuracy")
perf_train.accuracy()

perf_test = aml.leader.model_performance(test_hf)
print("test accuracy")
perf_test.accuracy()


# or if some subset of the models is needed a slice of leaderboard can be used, e.g., using MAE as the sorting metric
va_plot = h2o.varimp_heatmap(aml.leaderboard.head(1))

print(va_plot)

# or even extended leaderboard can be used
# va_plot = h2o.varimp_heatmap(h2o.automl.get_leaderboard(aml, extra_columns="training_time_ms").sort("training_time_ms").head(10))

exa = aml.leader.explain(train_hf)

# +

# # # Get model ids for all models in the AutoML Leaderboard
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# # Get the "All Models" Stacked Ensemble model
# se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in mid][0])
# # Get the Stacked Ensemble metalearner model
# se
# -

print(aml.leader.explain_row(test_hf, row_index=0))


my_local_model = h2o.download_model(aml.leader, path=models_path)
print(my_local_model)
# +
# metalearner = se.metalearner()
# metalearner

# +
# import os
# import re
# import shutil

# rootdir = models_path
# regex = re.compile('.*AutoML.*')

# for root, dirs, files in os.walk(rootdir):
#   for file in files:
#     if regex.match(file):

#         os.rename(models_path + file, models_path + "automl")

# +
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# GLM = h2o.get_model([mid for mid in model_ids if "GLM" in mid][0])
# XGBOOST = h2o.get_model([mid for mid in model_ids if "XGB" in mid][0])
# ENSEMBLE = h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in mid][0])
# models_to_save = [(GLM, "glm"), (XGBOOST, "xgboost"), (ENSEMBLE, "ensemble")]
