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
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer

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
    "R_ODDS",
    "B_ODDS",
]
random_seed = 1
cat_features = [
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
test_ratio = 0
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from mlxtend.data import mnist_data
from sklearn.metrics import accuracy_score, make_scorer
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

from sklearn.metrics import make_scorer
from sklearn.metrics import *


from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import accuracy_score
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression

random.seed(random_seed)

# %%
# D(P||Q) KL divergence, relative entropy of P, Q
def kl(P, Q):
    return (P * np.log(P / Q)).sum()

def st_kl(R, Q):
    return (R[R>0] * np.log(R[R>0] / Q[R>0])).sum()

prob_cols = ['R_DEC_P',
             'B_DEC_P',
             'R_SUB_P',
             'B_SUB_P',
             'R_KO_P',
             'B_KO_P']

odds_cols_alt = [
    "R_DEC_ODDS",
    "B_DEC_ODDS",
    "R_SUB_ODDS",
    "B_SUB_ODDS",
    "R_KO_ODDS",
    "B_KO_ODDS",
]

# %%
## GT
df = pd.read_csv(upstream['features-alt']['data'])
y_one_hot = df[target]
df['y_gt'] = df[target].values.argmax(axis = 1)

# %%
## Market
market = df[target + odds_cols_alt].dropna().reset_index(drop=True)
odds = market[odds_cols_alt].values
market_probs = sklearn.preprocessing.normalize(1 / odds, norm="l1")


market[prob_cols] = market_probs
market['y_mkt'] = market_probs.argmax(axis=1)
market['y_gt'] = market[target].values.argmax(axis = 1)

# %%
print("GT prior")
gt_prior = y_one_hot.sum(axis = 0) / len(y_one_hot)
print(gt_prior)

print("MKT prior")
print(target)
onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoded = onehot_encoder.fit_transform(market['y_mkt'].values.reshape(-1, 1)).astype(int)
mkt_prior = onehot_encoded.sum(axis = 0) / len(market.y_mkt)
print(mkt_prior)

# %%
print("market accuracy")
print((market.y_mkt == market.y_gt).sum() / len(market.y_mkt))
# %%


confmat = confusion_matrix(market.y_gt, market.y_mkt)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=False,
                                show_normed=True,
                                figsize=(4, 4))
plt.show()


# %%
X_fund = df.drop(columns = target)
if any(item in df.columns.tolist() for item in odds_cols):
    X_fund = X_fund.drop(columns=odds_cols)

X_fund = X_fund.dropna()
y = X_fund.y_gt.values

X_fund = X_fund.drop(columns = ['y_gt'])

num_features = X_fund.columns.difference(cat_features)

X_fund[num_features] = X_fund[num_features].astype(np.float64)
X_fund[cat_features] = X_fund[cat_features].astype(np.int64)

print(X_fund.columns)
print(X_fund.head())

X = X_fund.values

X = X.astype(np.float32)

X_train = X
y_train = y


# %%
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

clf7 = MLPClassifier(max_iter=10000,random_state=1)

clf8 = GradientBoostingClassifier(random_state=1)

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
                  ('std', StandardScaler()),
                  ('clf5', clf5)])

pipe6 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf6', clf6)])

pipe7 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('std', StandardScaler()),
                  ('clf7', clf7)])

pipe8 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf8', clf8)])

# Setting up the parameter grids
pg1 = [{'clf1__penalty': ['l2'],
                'clf1__C': np.power(10., np.arange(-4, 4))}]

pg2 = [{'clf2__n_neighbors': list(range(1, 10)),
        'clf2__p': [1, 2]}]

pg3 = [{'clf3__max_depth': list(range(1, 10)) + [None],
        'clf3__criterion': ['gini', 'entropy']}]

pg4 = [{'clf4__kernel': ['poly', 'rbf', 'sigmoid'],
        'clf4__degree': [1,2,3],
        'clf4__C': np.power(10., np.arange(-4, 4)),
        'clf4__gamma': np.power(10., np.arange(-5, 0))}]

pg5 = [{'clf5__n_estimators': [100, 500, 1000, 5000]}]


pg6 = [{'clf6__n_estimators': [100, 500, 1000, 5000],
        'clf6__criterion': ['gini', 'entropy'],
        # 'clf6__min_samples_leaf': param_range,
        # 'clf6__max_depth': param_range,
        # 'clf6__min_samples_split': param_range[1:]
        }]

pg7 = [{
        'clf7__hidden_layer_sizes': [(256,), (500,), (1000,), (500,2), (500, 10), (256,2), (256,10), (1000, 2)],
        'clf7__activation': ['tanh', 'relu'],
        'clf7__solver': ['sgd', 'adam'],
        'clf7__alpha': [0.0001, 0.05, 0.00047],
        'clf7__batch_size': ['auto'],
        'clf7__early_stopping': [True],
        'clf7__learning_rate': ['adaptive'],
        }]

pg8 = [{
    "clf8__loss":["deviance"],
    "clf8__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "clf8__min_samples_split": np.linspace(0.1, 0.5, 12),
    "clf8__min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "clf8__max_depth":[3,5,8],
    "clf8__max_features":["log2","sqrt"],
    "clf8__criterion": ["friedman_mse",  "mae"],
    "clf8__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "clf8__n_estimators":[10]
    }]




# %%
gridcvs = {}
inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1)

# zip((pg1, pg2, pg3, pg4),
# (pipe1, pipe2, pipe3, pipe4),
# ('lr', 'knn', 'dectree', 'svm')):

for pgrid, est, name in zip((pg1, pg2, pg3),
                            (pipe1, pipe2, pipe3),
                            ('lr', 'knn', 'dectree')):
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

def customLoss(xArray, yArray):
    return 5

scorer = {'score': 'accuracy',
        # 'custom': make_scorer(customLoss, greater_is_better=True),
        # 'cm': make_scorer(multilabel_confusion_matrix)
        # 'logloss': make_scorer(log_loss, labels=[0,1,2,3,4,5])
        }

dicts = {}
for name, gs_est in sorted(gridcvs.items()):
    scores_dict = cross_validate(gs_est,
                                 X=X_train,
                                 y=y_train,
                                 scoring=scorer,
                                 cv=outer_cv,
                                 return_estimator=True,
                                 n_jobs=6)

    print(50 * '-', '\n')
    print('Algorithm:', name)
    print('    Inner loop:')

    print(scores_dict['test_score'])
    dicts[name] = scores_dict

    for i in range(scores_dict['test_score'].shape[0]):
        print('\n        Best ACC (avg. of inner test folds) %.2f%%' % (scores_dict['estimator'][i].best_score_ * 100))
        print('        Best parameters:', scores_dict['estimator'][i].best_estimator_)
        print('        ACC (on outer test fold) %.2f%%' % (scores_dict['test_score'][i] * 100))

    print('\n%s | outer ACC %.2f%% +/- %.2f' %
          (name, scores_dict['test_score'].mean() * 100,
           scores_dict['test_score'].std() * 100))

# %%
# dummy
index = []
scores = {"Accuracy": [], "Balanced accuracy": []}

dummy_clf = DummyClassifier(strategy="most_frequent")
scoring = ["accuracy", "balanced_accuracy"]

index += ["Dummy classifier"]
cv_result = cross_validate(dummy_clf, X, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
num_pipe = make_pipeline(
    StandardScaler()
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0))



preprocessor_linear = make_column_transformer(
    (num_pipe, selector(dtype_include="float64")),
    (cat_pipe, selector(dtype_include="int64")),
    n_jobs=2,
)


lr_clf = make_pipeline(preprocessor_linear, LogisticRegression(max_iter=1000))

index += ["Logistic regression"]
cv_result = cross_validate(lr_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
# RF
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer


num_pipe = SimpleImputer(strategy="mean", add_indicator=True)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0),
)

preprocessor_tree = make_column_transformer(
    (num_pipe, selector(dtype_include="float64")),
    (cat_pipe, selector(dtype_include="int64")),
    n_jobs=2,
)

rf_clf = make_pipeline(
    preprocessor_tree, RandomForestClassifier(random_state=42, n_jobs=2)
)

index += ["Random forest cat"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
lr_clf.set_params(logisticregression__class_weight="balanced")

index += ["Logistic regression with balanced class weights"]
cv_result = cross_validate(lr_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
rf_clf.set_params(randomforestclassifier__class_weight="balanced")

index += ["Random forest with balanced class weights"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%


lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=42),
    LogisticRegression(max_iter=1000),
)

index += ["Under-sampling + Logistic regression"]
cv_result = cross_validate(lr_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
)
index += ["Under-sampling + Random forest"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%


rf_clf = make_pipeline(
    preprocessor_tree,
    BalancedRandomForestClassifier(random_state=42, n_jobs=2),
)

index += ["Balanced random forest"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores


# %%
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

bag_clf = make_pipeline(
    preprocessor_tree,
    BalancedBaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=2,
    ),
)

index += ["Balanced bag of histogram gradient boosting"]
cv_result = cross_validate(bag_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores


# %%
n = len(X_fund)
bound = int(n * 0.25)
X_train_df = X_fund.iloc[bound:-1,:]
X_test_df = X_fund.iloc[0:bound,:]

y_train = y[bound:-1]
y_test = y[0:bound]
# %%
from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1500, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# pipe5 = Pipeline([('prep', preprocessor_tree),
#                   ('sampler', RandomUnderSampler(random_state=1)),
#                   ('clf5', clf5)])

# pipe5 = make_pipeline_with_sampler(
#     preprocessor_tree,
#     RandomUnderSampler(random_state=1),
#     RandomForestClassifier(random_state=1, n_jobs=2))
# pipe5.set_params(randomforestclassifier__class_weight="balanced")


pipe5 = make_pipeline(
    preprocessor_tree,
    BalancedRandomForestClassifier(random_state=1, n_jobs=2),
    )



# Create the random grid
# random_grid = {'randomforestclassifier__n_estimators': n_estimators,
#                'randomforestclassifier__max_features': max_features,
#                'randomforestclassifier__max_depth': max_depth,
#                'randomforestclassifier__min_samples_split': min_samples_split,
#                'randomforestclassifier__min_samples_leaf': min_samples_leaf,
#                'randomforestclassifier__bootstrap': bootstrap}

random_grid = {'balancedrandomforestclassifier__n_estimators': n_estimators,
               'balancedrandomforestclassifier__max_features': max_features,
               'balancedrandomforestclassifier__max_depth': max_depth,
               'balancedrandomforestclassifier__min_samples_split': min_samples_split,
               'balancedrandomforestclassifier__min_samples_leaf': min_samples_leaf,
               'balancedrandomforestclassifier__bootstrap': bootstrap}
pprint(random_grid)



scorer = {'score': make_scorer(log_loss)}

# %%
rf_random = RandomizedSearchCV(scoring='neg_log_loss', estimator = pipe5, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1, refit=True)
# Fit the random search model
rf_random.fit(X_train_df, y_train)

# %%
rf_final = rf_random.best_estimator_
rf_final

# ('randomforestclassifier',
#  RandomForestClassifier(max_depth=90, max_features='sqrt',
#                         min_samples_leaf=4, n_estimators=1800,
#                         n_jobs=2, random_state=1))])

# %%

rf_final.fit(X_train_df, y_train)
y_hat = rf_final.predict(X_test_df)
confmat = confusion_matrix(y_test, y_hat)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=True,
                                show_normed=True,
                                figsize=(4, 4))
plt.show()
# %%
i = 0
for train_index, test_index in outer_cv.split(X, y):
    i += 1
    print({i})
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_hat_mkt = y_mkt[test_index]
    prob_mkt = mkt_probs[test_index]
    m = dicts['lr']['estimator'][0].best_estimator_
    m.fit(X_train, y_train)
    y_hat = m.predict(X_test)

    if i == 2:
        print("model")
        confmat = confusion_matrix(y_test, y_hat)
        fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                        show_absolute=True,
                                        show_normed=True,
                                        figsize=(4, 4))
        plt.show()

        print("market")
        confmat_mkt = confusion_matrix(y_test, y_hat_mkt)
        fig, ax = plot_confusion_matrix(conf_mat=confmat_mkt,
                                        show_absolute=True,
                                        show_normed=True,
                                        figsize=(4, 4))
        plt.show()


   # model.fit(X_train, y_train)
   # print confusion_matrix(y_test, model.predict(X_test))

# %%
n = int(X.shape[0] * 0.4)
X_train = X[n:]
X_test = X[0:n]

y_train = y[n:]
y_test = y[0:n]

y_train_mkt = y[n:]
# y_test_mkt = y_mkt[0:n]

# %%


rf_final.fit(X_train, y_train)
# %%
y_hat = rf_final.predict(X_test)
y_hat_probs = rf_final.predict_proba(X_test)
print("model")
confmat = confusion_matrix(y_test, y_hat)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=True,
                                show_normed=True,
                                figsize=(4, 4))
plt.show()

# %%

# print("market")
# confmat_mkt = confusion_matrix(y_test, y_hat_mkt)
# fig, ax = plot_confusion_matrix(conf_mat=confmat_mkt,
#                                 show_absolute=True,
#                                 show_normed=True,
#                                 figsize=(4, 4))
# plt.show()

# print("market")
# confmat_mkt = confusion_matrix(y_test, y_hat_mkt)
# fig, ax = plot_confusion_matrix(conf_mat=confmat_mkt,
#                                 show_absolute=True,
#                                 show_normed=True,
#                                 figsize=(4, 4))
# plt.show()



# %%
len(test_index)

#
#
# gridcvs = {}
# inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1)
#
# gcv = GridSearchCV(estimator=pipe5,
#                    param_grid=param_grid5,
#                    scoring='accuracy',
#                    n_jobs=-1,
#                    cv=inner_cv,
#                    verbose=0,
#                    refit=True)
# gridcvs['MLP_STD'] = gcv
#
# outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=1)
#
# for name, gs_est in sorted(gridcvs.items()):
#     scores_dict = cross_validate(gs_est,
#                                  X=X_train,
#                                  y=y_train,
#                                  cv=outer_cv,
#                                  return_estimator=True,
#                                  n_jobs=6)
#
#     print(50 * '-', '\n')
#     print('Algorithm:', name)
#     print('    Inner loop:')
#
#     for i in range(scores_dict['test_score'].shape[0]):
#         print('\n        Best ACC (avg. of inner test folds) %.2f%%' % (scores_dict['estimator'][i].best_score_ * 100))
#         print('        Best parameters:', scores_dict['estimator'][i].best_estimator_)
#         print('        ACC (on outer test fold) %.2f%%' % (scores_dict['test_score'][i] * 100))
#
#     print('\n%s | outer ACC %.2f%% +/- %.2f' %
#           (name, scores_dict['test_score'].mean() * 100,
#            scores_dict['test_score'].std() * 100))


# # %%
# onehot_encoder = OneHotEncoder(sparse=False)
#
# onehot_encoded = onehot_encoder.fit_transform(market['y_mkt'].values.reshape(-1, 1)).astype(int)


# your code here...
# -

