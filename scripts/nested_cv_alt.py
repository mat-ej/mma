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

# %% tags=["parameters"] trusted=true
upstream = None
product = None
target = None
random_seed = None
validation_ratio = None
odds_cols = None

# %% tags=["injected-parameters"] trusted=true
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



# %% trusted=false
import random
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_confusion_matrix
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
random.seed(random_seed)


# %% trusted=false
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

# %% trusted=false
## GT
df = pd.read_csv(upstream['features-alt']['data'])
y_one_hot = df[target]
df['y_gt'] = df[target].values.argmax(axis = 1)

# %% trusted=false
## Market
market = df[target + odds_cols_alt].dropna().reset_index(drop=True)
odds = market[odds_cols_alt].values
market_probs = sklearn.preprocessing.normalize(1 / odds, norm="l1")


market[prob_cols] = market_probs
market['y_mkt'] = market_probs.argmax(axis=1)
market['y_gt'] = market[target].values.argmax(axis = 1)

# %% trusted=false
print("GT prior")
gt_prior = y_one_hot.sum(axis = 0) / len(y_one_hot)
print(gt_prior)

print("MKT prior")
print(target)
onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoded = onehot_encoder.fit_transform(market['y_mkt'].values.reshape(-1, 1)).astype(int)
mkt_prior = onehot_encoded.sum(axis = 0) / len(market.y_mkt)
print(mkt_prior)

# %% trusted=false
from sklearn.metrics import balanced_accuracy_score

print("market accuracy")
print((market.y_mkt == market.y_gt).sum() / len(market.y_mkt))

balanced_accuracy_score(market.y_gt, market.y_mkt)
# %% trusted=false


confmat = confusion_matrix(market.y_gt, market.y_mkt)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=False,
                                show_normed=True,
                                # class_names=[1, 0],
                                figsize=(4, 4))
fig.tight_layout()
fig.savefig('fig/confmat_alt.eps', bbox_inches='tight', pad_inches=0)
plt.show()


# %% trusted=false
X_fund = df.drop(columns = target)
if any(item in df.columns.tolist() for item in odds_cols):
    X_fund = X_fund.drop(columns=odds_cols)

X_fund = X_fund.dropna()
y = X_fund.y_gt.values

X_fund = X_fund.drop(columns = ['y_gt'])

num_features = X_fund.columns.difference(cat_features)

X_fund[num_features] = X_fund[num_features].astype(np.float64)
X_fund[cat_features] = X_fund[cat_features].astype(np.int64)

X_fund['R_ODDS'] = df['R_ODDS']
X_fund['R_ODDS'] = X_fund['R_ODDS'].astype(np.float64)

X_fund['B_ODDS'] = df['B_ODDS']
X_fund['B_ODDS'] = X_fund['B_ODDS'].astype(np.float64)

print(X_fund.columns)
print(X_fund.head())

X = X_fund.values

X = X.astype(np.float32)

X_train = X
y_train = y


# %% trusted=false
# dummy
index = []
scores = {"Accuracy": [], "Balanced accuracy": [], "Negative log loss": []}

dummy_clf = DummyClassifier(strategy="most_frequent")
scoring = ["accuracy", "balanced_accuracy", "neg_log_loss"]

index += ["Dummy classifier"]
cv_result = cross_validate(dummy_clf, X, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false
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
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false
# RF
from sklearn.compose import make_column_selector as selector

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
    preprocessor_tree, RandomForestClassifier(n_jobs=2, max_depth=90, max_features='sqrt',min_samples_leaf=4, n_estimators=1800, random_state=random_seed)
)

index += ["Random forest cat"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false
lr_clf.set_params(logisticregression__class_weight="balanced")

index += ["Logistic regression with balanced class weights"]
cv_result = cross_validate(lr_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false
rf_clf.set_params(randomforestclassifier__class_weight="balanced")

index += ["Random forest with balanced class weights"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false


lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=1),
    LogisticRegression(max_iter=1000),
)

index += ["Under-sampling + Logistic regression"]
cv_result = cross_validate(lr_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false
rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    RandomUnderSampler(random_state=1),
    RandomForestClassifier(n_jobs=2, max_depth=90, max_features='auto',min_samples_leaf=4, n_estimators=1800, random_state=random_seed)
)
index += ["Under-sampling + Random forest"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% trusted=false


rf_clf = make_pipeline(
    preprocessor_tree,
    BalancedRandomForestClassifier(bootstrap=False,
                                   max_depth=30,
                                   min_samples_leaf=4,
                                   n_estimators=1250, n_jobs=2,
                                   random_state=1)
)

index += ["Balanced random forest"]
cv_result = cross_validate(rf_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores


# %% trusted=false

bag_clf = make_pipeline(
    preprocessor_tree,
    BalancedBaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=1000,
        random_state=42,
        n_jobs=2,
    ),
)

index += ["Balanced bag of histogram gradient boosting"]
cv_result = cross_validate(bag_clf, X_fund, y, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
scores["Negative log loss"].append(cv_result["test_neg_log_loss"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores


# %% trusted=false
n = len(X_fund)
bound = int(n * 0.25)
X_train_df = X_fund.iloc[bound:-1,:]
X_test_df = X_fund.iloc[0:bound,:]

y_train = y[bound:-1]
y_test = y[0:bound]
# %% trusted=false
from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1500, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
max_features_bag = np.linspace(0,1, 10)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
bootstrap_features = [True, False]

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


bag_clf = make_pipeline(
    preprocessor_tree,
    BalancedBaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(random_state=1),
        n_estimators=500,
        random_state=42,
        n_jobs=2,
    ),
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


bag_grid = {'balancedbaggingclassifier__n_estimators': n_estimators,
            'balancedbaggingclassifier__max_features': max_features_bag,
            'balancedbaggingclassifier__bootstrap': bootstrap,
            'balancedbaggingclassifier__bootstrap_features': bootstrap_features
            }
pprint(random_grid)



scorer = {'score': make_scorer(log_loss)}

# %% trusted=false
rf_random = RandomizedSearchCV(scoring='neg_log_loss', estimator = bag_clf, param_distributions = bag_grid, n_iter = 100, cv = 3, verbose=3, random_state=1, n_jobs = -1, refit=True)
# Fit the random search model
rf_random.fit(X_train_df, y_train)

# %% trusted=false
rf_final = rf_random.best_estimator_
rf_final

# ('randomforestclassifier',
#  RandomForestClassifier(max_depth=90, max_features='sqrt',
#                         min_samples_leaf=4, n_estimators=1800,
#                         n_jobs=2, random_state=1))])

# BalancedRandomForestClassifier(bootstrap=False, max_depth=30,
#                                                 min_samples_leaf=4,
#                                                 n_estimators=1250, n_jobs=2,
#                                                 random_state=1))]

# %% trusted=false

rf_final.fit(X_train_df, y_train)
y_hat = rf_final.predict(X_test_df)
confmat = confusion_matrix(y_test, y_hat)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=True,
                                show_normed=True,
                                figsize=(4, 4))
plt.show()

# %% trusted=false
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




# %% trusted=false
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


# %% trusted=false
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

# %% trusted=false
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

# %% trusted=false
n = int(X.shape[0] * 0.4)
X_train = X[n:]
X_test = X[0:n]

y_train = y[n:]
y_test = y[0:n]

y_train_mkt = y[n:]
# y_test_mkt = y_mkt[0:n]

# %% trusted=false


rf_final.fit(X_train, y_train)
# %% trusted=false
y_hat = rf_final.predict(X_test)
y_hat_probs = rf_final.predict_proba(X_test)
print("model")
confmat = confusion_matrix(y_test, y_hat)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=True,
                                show_normed=True,
                                figsize=(4, 4))
plt.show()

# %% trusted=false

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



# %% trusted=false
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

