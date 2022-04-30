# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# %% tags=["parameters"]
upstream = None
product = None
target = None
random_seed = None
validation_ratio = None
odds_cols = None

# %% tags=["injected-parameters"]
# Parameters
target = ["R_DEC", "B_DEC", "R_SUB", "B_SUB", "R_KO", "B_KO"]
odds_cols = [
    "R_DEC_ODDS",
    "B_DEC_ODDS",
    "R_SUB_ODDS",
    "B_SUB_ODDS",
    "R_KO_ODDS",
    "B_KO_ODDS",
]
random_seed = 1
test_ratio = 0.1
inner_splits = 2
outer_splits = 5
upstream = {
    "features-alt": {
        "data": "/home/m/repo/mma/products/data/features_odds_alt.csv",
        "nb": "/home/m/repo/mma/products/reports/features_odds_alt.ipynb",
    }
}
product = {
    "nb": "/home/m/repo/mma/products/reports/nested_cv_alt.ipynb",
    "model": "/home/m/repo/mma/products/models/nested_cv_alt.pt",
}


# %%
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from mlxtend.data import mnist_data
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from pprint import pprint
import PipelineProfiler
from sklearn.pipeline import Pipeline
import autosklearn
from sklearn.decomposition import FastICA

from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
from sklearn.neural_network import MLPClassifier
import sklearn
from mlxtend.evaluate import confusion_matrix


from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import accuracy_score
import matplotlib.pyplot as plt
import random
random.seed(random_seed)
# %%
df = pd.read_csv(upstream['features-alt']['data']).dropna().reset_index()
market = df[odds_cols]
y_one_hot = df[target]
y_true = df[target].values.argmax(axis = 1)

market['y_true'] = y_true

#%%

odds = df[odds_cols].values
market_probs = sklearn.preprocessing.normalize(1 / odds, norm="l1")

prob_cols = ['R_DEC_P',
             'B_DEC_P',
             'R_SUB_P',
             'B_SUB_P',
             'R_KO_P',
             'B_KO_P']

market[prob_cols] = market_probs
market['y_mkt'] = market_probs.argmax(axis=1)


print("GT prior")
gt_prior = y_one_hot.sum(axis = 0) / len(y_one_hot)
print(gt_prior)

print("MKT prior")
print(target)
onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoded = onehot_encoder.fit_transform(market['y_mkt'].values.reshape(-1, 1)).astype(int)
mkt_prior = onehot_encoded.sum(axis = 0) / len(y_true)
print(mkt_prior)

# %%
# (market.y_mkt == market.y_true).sum() / len(y_true)


y_mkt = market.y_mkt.values

print((y_mkt == y_true).sum() / len(y_mkt))
#%%



confmat = confusion_matrix(y_true, market.y_mkt)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=False,
                                show_normed=True,
                                figsize=(4, 4))
plt.show()


# %%
# D(P||Q) KL divergence, relative entropy of P, Q
def kl(P, Q):
    return (P * np.log(P / Q)).sum()

def st_kl(R, Q):
    return (R[R>0] * np.log(R[R>0] / Q[R>0])).sum()




X_fund = df.drop(columns = target)

if odds_cols in df.columns.tolist():
    X_fund = X_fund.drop(columns=odds_cols)

if any(item in df.columns.tolist() for item in odds_cols):
    X_fund = X_fund.drop(columns=odds_cols)

print(X_fund.columns)

print(X_fund.head())
# print(y_true.head())



X = X_fund.values
y = y_true

X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_ratio,
                                                    random_state=random_seed,
                                                    stratify=y)

#%%

clf5 = RandomForestClassifier(random_state=1)
pipe5 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf5', clf5)])

param_grid5 = [{'clf5__n_estimators': [100, 500, 1000, 5000]}]


gridcvs = {}
inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1)

gcv = GridSearchCV(estimator=pipe5,
                   param_grid=param_grid5,
                   scoring='accuracy',
                   n_jobs=-1,
                   cv=inner_cv,
                   verbose=0,
                   refit=True)
gridcvs['MLP_STD'] = gcv

outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=1)

for name, gs_est in sorted(gridcvs.items()):
    scores_dict = cross_validate(gs_est,
                                 X=X_train,
                                 y=y_train,
                                 cv=outer_cv,
                                 return_estimator=True,
                                 n_jobs=6)

    print(50 * '-', '\n')
    print('Algorithm:', name)
    print('    Inner loop:')

    for i in range(scores_dict['test_score'].shape[0]):
        print('\n        Best ACC (avg. of inner test folds) %.2f%%' % (scores_dict['estimator'][i].best_score_ * 100))
        print('        Best parameters:', scores_dict['estimator'][i].best_estimator_)
        print('        ACC (on outer test fold) %.2f%%' % (scores_dict['test_score'][i] * 100))

    print('\n%s | outer ACC %.2f%% +/- %.2f' %
          (name, scores_dict['test_score'].mean() * 100,
           scores_dict['test_score'].std() * 100))


# # %%
# onehot_encoder = OneHotEncoder(sparse=False)
#
# onehot_encoded = onehot_encoder.fit_transform(market['y_mkt'].values.reshape(-1, 1)).astype(int)


# your code here...
# -

