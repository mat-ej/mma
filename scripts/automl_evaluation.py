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
# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
upstream = {
    "automl-sklearn": {
        "nb": "/home/m/repo/mma/products/reports/fit-sklearn-automl.ipynb",
        "model": "/home/m/repo/mma/products/models/sklearn-automl.pickle",
    },
    "automl-h2o": {"nb": "/home/m/repo/mma/products/reports/fit_h2o_automl.ipynb"},
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    },
}
product = {"nb": "/home/m/repo/mma/products/reports/automl_evaluation.ipynb"}

# -

# target_win = ['WINNER']
# names = ['R_NAME', 'B_NAME']
# dates = ['DATE']
# target_win_odds = ['R_ODDS', 'B_ODDS']
# target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
# target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']
#
#
import pandas as pd
import numpy as np
import pickle
import PipelineProfiler

test_df = pd.read_csv(upstream['split-train-test']['test'])
train_df = pd.read_csv(upstream['split-train-test']['train'])
automl_sklearn = pickle.load(open(upstream['automl-sklearn']['model'], 'rb'))
print(automl_sklearn)
print(test_df)

print("AUTOSKLEARN PIPELINE")
profiler_data = PipelineProfiler.import_autosklearn(automl_sklearn)
PipelineProfiler.plot_pipeline_matrix(profiler_data)

X_test = test_df.drop(columns = target)
Y_test = test_df[target]

X_train = train_df.drop(columns = target)
Y_train = train_df[target]

if len(target) == 1:
    y_train = Y_train[target[0]].values
    y_hat_train = automl_sklearn.predict(X_train)
    print("train accuracy")
    print(np.sum(y_hat_train == y_train) / len(y_hat_train))

    y_test = Y_test[target[0]].values
    y_hat = automl_sklearn.predict(X_test)
    print("test accuracy")
    print(np.sum(y_hat == y_test) / len(y_hat))

# X_test = test_df[X_columns]
# Y_test = test_df['WINNER'].values
# Y_model_probs = model.predict_proba(X_test)
# Y_model = model.predict(X_test)
# model_acc = np.sum(Y_test == Y_model) / len(Y_test)
# print(model_acc)

# odds = test_df[target_win_odds]
# Y_market = (odds.R_ODDS < odds.B_ODDS).values
# market_acc = np.sum(Y_test == Y_market) / len(Y_test)
#
# print(market_acc)
#
#




