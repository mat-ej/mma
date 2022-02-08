
target = ["WINNER"]
random_seed = 1
autosklearn_config = {
    "n_jobs": 4,
    "memory_limit": 1024,
    "time_left_for_this_task": 30,
    "per_run_time_limit": 10,
}
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

import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics

import sklearn
from sklearn.metrics import fbeta_score, make_scorer
import numpy as np


def accuracy_wk(solution, prediction, extra_argument):
    # custom function defining accuracy and accepting an additional argument
    print(extra_argument)
    return np.mean(solution == prediction)

accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu_add",
    score_func=accuracy_wk,
    optimum=0,
    greater_is_better=False,
    needs_proba=True,
    needs_threshold=False,
    extra_argument=X_train,
)

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    per_run_time_limit=10,
    seed=1,
    metric=accuracy_scorer
)
cls.fit(X_train, Y_train)

predictions = cls.predict(X_test)
score = accuracy_scorer(Y_test, predictions)
metric_name = cls.automl_._metric.name

print(f"Accuracy score {score:.3f} using {metric_name:s}")