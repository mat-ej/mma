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
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
print("market accuracy")
print((market.y_mkt == market.y_gt).sum() / len(market.y_mkt))
# %% trusted=false


confmat = confusion_matrix(market.y_gt, market.y_mkt)
fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=False,
                                show_normed=True,
                                figsize=(4, 4))
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
    RandomUnderSampler(random_state=42),
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
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
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
        n_estimators=500,
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
        random_state=1,
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
