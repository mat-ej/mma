# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# +
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://ploomber.readthedocs.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# + tags=["parameters"]
# extract_upstream=False in your pipeline.yaml file, if this task has
# dependencies, declare them in the YAML spec and leave this as None. Once you
# add the dependencies, reload the file for Ploomber to inject the cell
# (On JupyterLab: File -> Reload File from Disk)
import pickle

import numpy as np
from autosklearn.classification import AutoSklearnClassifier

upstream = None

# extract_product=False in your pipeline.yaml file, leave this as None, the
# value in the YAML spec will be injected in a cell below. If you don't see it,
# check the Jupyter logs
product = None


# + tags=["injected-parameters"]
# Parameters
upstream = {"split-train-test": {"train": "/home/m/repo/mma/products/data/train.csv", "test": "/home/m/repo/mma/products/data/test.csv"}}
product = {"nb": "/home/m/repo/mma/products/reports/fit-sklearn-automl.ipynb", "model": "/home/m/repo/mma/products/models/sklearn-automl.pickle"}


# +
import pandas as pd

target_win = ['WINNER']
names = ['R_NAME', 'B_NAME']
dates = ['DATE']
target_win_odds = ['R_ODDS', 'B_ODDS']
target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']

#

# df_clean = df[X_columns + target_win].dropna()
# train_df, test_df = split_train_test(df_clean, 0.3)

train_df = pd.read_csv(upstream['split-train-test']['train'], parse_dates=['DATE'])
test_df = pd.read_csv(upstream['split-train-test']['test'], parse_dates=['DATE'])

print(train_df.columns)

X_columns = train_df.columns.difference(target_win + target_win_odds + target + target_odds + names + dates).to_list()

train_df = train_df[X_columns + target_win].dropna()
test_df = test_df[X_columns + target_win].dropna()

X_train = train_df[X_columns]
Y_train = train_df[target_win]

X_test = test_df[X_columns]
Y_test = test_df[target_win]

automl = AutoSklearnClassifier(
        # time_left_for_this_task=30,
        # per_run_time_limit=10,
        # tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
        n_jobs=7,
        # Each one of the 4 jobs is allocated 3GB
        memory_limit=1024,
        seed=5,
)

automl.fit(X_train, Y_train, dataset_name='mma')

print("Statistics")
print(automl.sprint_statistics())
print(automl.show_models())
y_hat = automl.predict(X_test)

print(y_hat)

print("accuracy")
# print(np.sum(y_hat == Y_test) / len(y_hat))

# x = automl.show_models()
# print(x)

with open(product['model'], 'wb') as f:
    pickle.dump(automl, f)

