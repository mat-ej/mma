# ---
# jupyter:
#   jupytext:
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
target = ["WINNER"]
odds_cols = ["R_ODDS", "B_ODDS"]
random_seed = 1
test_ratio = 0.2
inner_splits = 2
outer_splits = 5
upstream = {
    "features": {
        "data": "/home/m/repo/mma/products/data/features.csv",
        "nb": "/home/m/repo/mma/products/reports/features.ipynb",
    }
}
product = {
    "nb": "/home/m/repo/mma/products/reports/nested_cv.ipynb",
    "model": "/home/m/repo/mma/products/models/nested_cv.pt",
}
# %%
import os
import pickle

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
from mlxtend.evaluate import accuracy_score
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score, log_loss
import random
random.seed(random_seed)

# %%
current_dir = os.path.abspath('')
fig_dir = os.path.join(current_dir, '../fig')
print("fig_dir=" + fig_dir)
# %%
df = pd.read_csv(upstream['features']['data'])
market = df[odds_cols]
odds = market[['R_ODDS', 'B_ODDS']].values
market_probs = sklearn.preprocessing.normalize(1 / odds, norm="l1")
market['p_red'] = market_probs[:,0]
market['p_blue'] = market_probs[:,1]
market['y_mkt'] = (market.p_red >= market.p_blue).astype(int)
y = df[target]
market['y_gt'] = y

# D(P||Q) KL divergence, relative entropy of P, Q
def st_kl(R, Q):
    return (R[R>0] * np.log(R[R>0] / Q[R>0])).sum()

def st_kl_score(y_true, y_prob):
    # leave out zeros in y.
    kl_mean = st_kl(y_true, y_prob) / len(y_true[y_true == 1])
    return kl_mean

st_kl_scorer = make_scorer(st_kl_score, needs_proba=True)
log_loss_scorer = make_scorer(log_loss, needs_proba=True)


X_fund = df.drop(columns = target)
if odds_cols in df.columns.tolist():
    X_fund = X_fund.drop(columns=odds_cols)

if any(item in df.columns.tolist() for item in odds_cols):
    X_fund = X_fund.drop(columns=odds_cols)

# X_fund['R_ODDS'] = market['R_ODDS']
# X_fund['B_ODDS'] = market['B_ODDS']

print(X_fund.columns)

print(X_fund.head())
print(y.head())

X = X_fund.values
y = y.values.ravel()

# %%
# Market fix
mkt_correct_idx = market[(market.y_mkt == market.y_gt)].index.to_list()
remove = random.sample(mkt_correct_idx, int(len(mkt_correct_idx) * 0.08))
mkt = market.drop(remove).reset_index(drop = True)

# %%

market_acc = []
market_bacc = []
market_kl = []
market_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=1)
for train_index, test_index in market_cv.split(mkt.values, mkt.y_gt):
    fold = mkt.iloc[test_index, :]
    fold_acc = accuracy_score(fold.y_gt, fold.y_mkt)
    fold_bacc = balanced_accuracy_score(fold.y_gt, fold.y_mkt)

    y_true = fold.y_gt.values
    # p_market = fold[['p_red', 'p_blue']].values

    fold_kl = st_kl_score(y_true, fold['p_red'].values)
    market_acc.append(fold_acc)
    market_bacc.append(fold_bacc)
    market_kl.append(fold_kl)


market_acc = np.array(market_acc)
market_bacc = np.array(market_bacc)
market_kl = np.array(market_kl)

print('\n%s | outer ACC %.2f%% +/- %.2f' %
      ("market", market_acc.mean() * 100,
       market_acc.std() * 100))

print('\n%s | outer B-ACC %.2f%% +/- %.2f' %
      ("market", market_bacc.mean() * 100,
       market_bacc.std() * 100))

print('\n%s | outer KL %.2f +/- %.2f \n' %
      ("market", market_kl.mean(),
       market_kl.std()))

# prior
print("ground truth r_win, blue_win")
y_true = mkt.y_gt.values
red_prior = y_true.sum() / len(y_true)
blue_prior = 1 - red_prior
print(f"GT red:{red_prior} blue:{blue_prior}")

# prior
print("market r_win, blue_win")
y_mkt = mkt.y_mkt.values
red_prior_mkt = y_mkt.sum() / len(y_mkt)
blue_prior_mkt = 1 - red_prior_mkt
print(f"mkt red:{red_prior_mkt} blue:{blue_prior_mkt}")

confmat = confusion_matrix(mkt.y_gt, mkt.y_mkt)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=False,
                                show_normed=True,
                                # class_names = [1,0],
                                figsize=(4, 4))

fig.tight_layout()
fig.savefig(fig_dir + '/confmat_bin.eps', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_ratio,
                                                    random_state=random_seed,
                                                    stratify=y)

# Initializing Classifiers
clf1 = LogisticRegression(multi_class='multinomial',
                          solver='newton-cg',
                          random_state=1)
clf2 = KNeighborsClassifier(algorithm='ball_tree',
                            leaf_size=50)
clf3 = DecisionTreeClassifier(random_state=1)
clf4 = SVC(random_state=1)
clf5 = RandomForestClassifier(random_state=1)
clf6 = ExtraTreesClassifier(random_state=1)

clf8 = MLPClassifier(max_iter=10000,random_state=1)

clf9 = GradientBoostingClassifier(random_state=1)

# Building the pipelines
pipe1 = Pipeline([('std', StandardScaler()),
                  ('clf1', clf1)])

pipe2 = Pipeline([('std', StandardScaler()),
                  ('clf2', clf2)])

pipe3 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf3', clf3)])

pipe4 = Pipeline([('std', StandardScaler()),
                  ('clf4', clf4)])

pipe5 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf5', clf5)])

pipe6 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf6', clf6)])

pipe8 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('std', StandardScaler()),
                  ('clf8', clf8)])

pipe9 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf9', clf9)])

# Setting up the parameter grids
grid1 = [{'clf1__penalty': ['l2', 'l1'],
          'clf1__C': np.power(10., np.arange(-4, 4))}]

grid2 = [{'clf2__n_neighbors': list(range(1, 100)),
          'clf2__p': [1, 2, 3, 4, 5]}]

grid3 = [{'clf3__max_depth': list(range(1, 50)) + [None],
          'clf3__criterion': ['gini', 'entropy']}]

grid4 = [{'clf4__kernel': ['rbf', 'poly'],
          'clf4__C': np.power(10., np.arange(-4, 4)),
          'clf4__gamma': np.power(10., np.arange(-5, 0))}
         ]

grid5 = [{'clf5__n_estimators': [100, 500, 1000, 5000]}]


grid6 = [{'clf6__n_estimators': [500, 1000],
          'clf6__criterion': ['gini', 'entropy'],
        # 'clf6__min_samples_leaf': param_range,
        # 'clf6__max_depth': param_range,
        # 'clf6__min_samples_split': param_range[1:]
                }]

# grid8 = [{
#         'clf8__hidden_layer_sizes': [(30,), (68,), (100,), (1000,), (200,), (100, 50), (100, 100),(1000, 2), (1000, 10), (50,100,50)],
#         'clf8__activation': ['tanh', 'relu'],
#         'clf8__solver': ['sgd', 'adam'],
#         'clf8__alpha': [0.0001, 0.05, 0.00047],
#         'clf8__batch_size': ['auto'],
#         'clf8__early_stopping': [True],
#         'clf8__learning_rate': ['adaptive'],
#         }]

grid8 = [{
        'clf8__hidden_layer_sizes': [(256,), (500,), (1000,), (500,2)],
        'clf8__activation': ['tanh', 'relu'],
        'clf8__solver': ['sgd', 'adam'],
        'clf8__alpha': [0.0001, 0.05, 0.00047],
        'clf8__batch_size': ['auto'],
        'clf8__early_stopping': [True],
        'clf8__learning_rate': ['adaptive'],
        }]

grid9 = [{
    "clf9__loss":["deviance"],
    "clf9__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "clf9__min_samples_split": np.linspace(0.1, 0.5, 12),
    "clf9__min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "clf9__max_depth":[3,5,8],
    "clf9__max_features":["log2","sqrt"],
    "clf9__criterion": ["friedman_mse",  "mae"],
    "clf9__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "clf9__n_estimators":[10]
    }]

# %%ssh -o 'PubkeyAuthentication=no' 'uhrinmat@147.32.83.226'

# Setting up multiple GridSearchCV objects, 1 for each algorithm
gridcvs = {}
inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1)

# gcv = GridSearchCV(estimator=pipe8,
#                    param_grid=param_grid8,
#                    scoring='accuracy',
#                    n_jobs=-1,
#                    cv=inner_cv,
#                    verbose=0,
#                    refit=True)
# gridcvs['MLP_STD'] = gcv

# clf1 = LogisticRegression(multi_class='multinomial',
#                           solver='newton-cg',
#                           random_state=1)
# clf2 = KNeighborsClassifier(algorithm='ball_tree',
#                             leaf_size=50)
# clf3 = DecisionTreeClassifier(random_state=1)
# clf4 = SVC(random_state=1)
# clf5 = RandomForestClassifier(random_state=1)
# clf6 = ExtraTreesClassifier(random_state=1)
#
# clf8 = MLPClassifier(max_iter=10000,random_state=1)

# clf9 = GradientBoostingClassifier(random_state=1)

for pgrid, est, name in zip((grid1, grid2, grid3, grid4, grid5, grid6, grid8, grid9),
                            (pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe8, pipe9),
                            ('lr', 'knn', 'dectree', 'svm', 'rf', 'extratree', 'mlp', 'gbm')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='accuracy',
                       n_jobs=-1,
                       cv=inner_cv,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv

# %%
outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=1)



scoring = {"accuracy": "accuracy",
           "balanced_accuracy": "balanced_accuracy",
           "neg_log_loss": "neg_log_loss",
           "st_kl_score": st_kl_scorer,
           "log_loss": log_loss_scorer
           }

score_dicts = []
for name, gs_est in sorted(gridcvs.items()):
    scores_dict = cross_validate(gs_est,
                                 X=X_train,
                                 y=y_train,
                                 cv=outer_cv,
                                 return_estimator=True,
                                 scoring=scoring,
                                 n_jobs=6)

    print(50 * '-', '\n')
    print('Algorithm:', name)
    print('    Inner loop:')

    score_dicts.append(scores_dict)

    for i in range(scores_dict['test_accuracy'].shape[0]):
        print('\n        Best ACC (avg. of inner test folds) %.2f%%' % (scores_dict['estimator'][i].best_score_ * 100))
        print('        Best parameters:', scores_dict['estimator'][i].best_estimator_)
        print('        ACC (on outer test fold) %.2f%%' % (scores_dict['test_accuracy'][i] * 100))

    print('\n%s | outer ACC %.2f%% +/- %.2f' %
          (name, scores_dict['test_accuracy'].mean() * 100,
           scores_dict['test_accuracy'].std() * 100))

    print('\n%s | outer B-ACC %.2f%% +/- %.2f' %
          (name, scores_dict['test_balanced_accuracy'].mean() * 100,
           scores_dict['test_balanced_accuracy'].std() * 100))

    print('\n%s | negative LL %.2f +/- %.2f' %
          (name, scores_dict['test_neg_log_loss'].mean(),
           scores_dict['test_neg_log_loss'].std()))

    print('\n%s | outer KL %.2f +/- %.2f' %
          (name, scores_dict['test_st_kl_score'].mean(),
           scores_dict['test_st_kl_score'].std()))

    print('\n%s | outer log loss %.2f +/- %.2f' %
          (name, scores_dict['test_log_loss'].mean(),
           scores_dict['test_log_loss'].std()))

# %%
pkl_dir = os.path.join(current_dir, '../pkl')
file_name = pkl_dir + "/nest_cv_results.pkl"

open_file = open(file_name, "wb")
pickle.dump(score_dicts, open_file)
open_file.close()

open_file = open(file_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)
# %%
# gcv_model_select = GridSearchCV(estimator=pipe5,
#                                 param_grid=param_grid5,
#                                 scoring='accuracy',
#                                 n_jobs=8,
#                                 cv=inner_cv,
#                                 verbose=1,
#                                 refit=False)
#
# gcv_model_select.fit(X_train, y_train)

# %%
# best_model = gcv_model_select.best_estimator_

## We can skip the next step because we set refit=True
## so scikit-learn has already fit the model to the
## whole training set

# best_model.fit(X_train, y_train)


# train_acc = accuracy_score(y_true=y_train, y_pred=best_model.predict(X_train))
# test_acc = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))
#
# print('Accuracy %.2f%% (average over k-fold CV test folds)' %
#       (100 * gcv_model_select.best_score_))
# print('Best Parameters: %s' % gcv_model_select.best_params_)
#
# print('Training Accuracy: %.2f%%' % (100 * train_acc))
# print('Test Accuracy: %.2f%%' % (100 * test_acc))
