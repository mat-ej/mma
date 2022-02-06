import multiprocessing
import subprocess
import time

from dask.distributed import Client
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
product = {"nb": "/home/m/repo/mma/products/reports/fit-simple.ipynb", "model": "/home/m/repo/mma/products/models/simple-model.pickle"}


if __name__ == '__main__':

    import pandas as pd
    target_win = ['WINNER']
    names = ['R_NAME', 'B_NAME']
    dates = ['DATE']
    target_win_odds = ['R_ODDS', 'B_ODDS']
    target = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB']
    target_odds = ['R_DEC_ODDS', 'R_KO_ODDS', 'R_SUB_ODDS', 'B_DEC_ODDS', 'B_KO_ODDS', 'B_SUB_ODDS']


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

    client = Client(processes = False)

    print(client)
    automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            # per_run_time_limit=10,
            # tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
            # n_jobs=7,
            # # Each one of the 4 jobs is allocated 3GB
            # memory_limit=1024,
            # seed=5,
            dask_client=client
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

