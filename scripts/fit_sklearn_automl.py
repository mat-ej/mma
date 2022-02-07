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
from dask.distributed import Client



# extract_product=False in your pipeline.yaml file, leave this as None, the
# value in the YAML spec will be injected in a cell below. If you don't see it,
# check the Jupyter logs
upstream = None
product = None
target = None
random_seed = None

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
random_seed = 1
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    }
}
product = {
    "nb": "/home/m/repo/mma/products/reports/fit-sklearn-automl.ipynb",
    "model": "/home/m/repo/mma/products/models/sklearn-automl.pickle",
}


# +
import pandas as pd
train_df = pd.read_csv(upstream['split-train-test']['train'])
test_df = pd.read_csv(upstream['split-train-test']['test'])

X_train = train_df.drop(columns = target)
Y_train = train_df[target]

X_test = test_df.drop(columns = target)
Y_test = test_df[target]

print(train_df.columns)

client = Client(processes = False)

# if debug
# client = client(processes = False)
automl = AutoSklearnClassifier(
        # time_left_for_this_task=30,
        # per_run_time_limit=10,
        # tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
        # n_jobs=7,
        # Each one of the 4 jobs is allocated 3GB
        # memory_limit=1024,
        seed=random_seed,
        dask_client=client
)

automl.fit(X_train, Y_train, dataset_name='mma')

print("Statistics")
print(automl.sprint_statistics())
print(automl.show_models())

if len(target) == 1:
    y_test = Y_test[target[0]].values
    y_hat = automl.predict(X_test)
    print("test accuracy")
    print(np.sum(y_hat == y_test) / len(y_hat))

    y_train = Y_train[target[0]].values
    y_hat_train = automl.predict(X_train)
    print("train accuracy")
    print(np.sum(y_hat_train == y_train) / len(y_hat_train))

# print(automl.show_models())
#
with open(product['model'], 'wb') as f:
    pickle.dump(automl, f)

