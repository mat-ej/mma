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
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from mlxtend.data import mnist_data
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from pprint import pprint
import PipelineProfiler
from sklearn.pipeline import Pipeline
import autosklearn

from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing


# %%
df = pd.read_csv(upstream['features']['data'])

y = df[target]
X_fund = df.drop(columns = target)
if odds_cols in df.columns.tolist():
    X_fund = df.drop(columns=odds_cols)

print(X_fund.columns)

print(X_fund.head())
print(y.head())

X = X_fund.values
y = y.values.ravel()

# %%
automl_sklearn = pickle.load(open('/home/m/repo/mma/backup/sklearn-automl_no_odds.pickle', 'rb'))
# print(automl_sklearn)
# print(test_df)
print("Statistics")
print(automl_sklearn.sprint_statistics())
# print(automl.show_models())
pprint(automl_sklearn.show_models(), indent=4)
# df_cv_results = pd.DataFrame(automl_sklearn).sort_values(by = 'mean_test_score', ascending = False)
# print(df_cv_results)
# %%
leaderboard = automl_sklearn.leaderboard(detailed = True, ensemble_only=False)
automl_models = automl_sklearn.get_models_with_weights()
best_model = automl_models[0][1]

transform_pip = Pipeline([('data_pre', automl_sklearn.get_models_with_weights()[0][1][0]),
                         ('feature_pre', automl_sklearn.get_models_with_weights()[0][1][1])])

estimator = automl_sklearn.get_models_with_weights()[0][1][2]
# %%
# best pipeline

# balancing = Balancing(random_state=1, strategy='weighting')
# preprocessor = ExtraTreesPreprocessorClassification(
# bootstrap = False,
# criterion = 'entropy',
# max_depth = None,
# max_features = 0.99299,
# max_leaf_nodes = None,
# min_impurity_decrease = 0,
# min_samples_leaf  = 1,
# min_samples_split = 2,
# min_weight_fraction_leaf = 0,
# n_estimators = 100,
# )

# classifier = ExtraTreesClassifier

# for name in FeaturePreprocessorChoice.get_components():
#     print(name)

from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice

# %%
# Loading and splitting the dataset
# Note that this is a small (stratified) subset
# of MNIST; it consists of 5000 samples only, that is,
# 10% of the original MNIST dataset
# http://yann.lecun.com/exdb/mnist/
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

# Building the pipelines
pipe1 = Pipeline([('std', StandardScaler()),
                  ('clf1', clf1)])

pipe2 = Pipeline([('std', StandardScaler()),
                  ('clf2', clf2)])

pipe4 = Pipeline([('std', StandardScaler()),
                  ('clf4', clf4)])

pipe5 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf5', clf5)])

pipe6 = Pipeline([('balancing', Balancing(random_state=1, strategy='weighting')),
                  ('clf6', clf6)])


# Setting up the parameter grids
param_grid1 = [{'clf1__penalty': ['l2'],
                'clf1__C': np.power(10., np.arange(-4, 4))}]

param_grid2 = [{'clf2__n_neighbors': list(range(1, 10)),
                'clf2__p': [1, 2]}]

param_grid3 = [{'max_depth': list(range(1, 10)) + [None],
                'criterion': ['gini', 'entropy']}]

param_grid4 = [{'clf4__kernel': ['rbf'],
                'clf4__C': np.power(10., np.arange(-4, 4)),
                'clf4__gamma': np.power(10., np.arange(-5, 0))},
               {'clf4__kernel': ['linear'],
                'clf4__C': np.power(10., np.arange(-4, 4))}]

param_grid5 = [{'clf5__n_estimators': [100, 500, 1000, 10000]}]

param_range = list(range(0, 20))
param_grid6 = [{'clf6__n_estimators': [100, 500, 1000, 10000],
                'clf6__criterion': ['gini', 'entropy'],
                # 'clf6__min_samples_leaf': param_range,
                # 'clf6__max_depth': param_range,
                # 'clf6__min_samples_split': param_range[1:]
                }
               ]

# %%
# Setting up multiple GridSearchCV objects, 1 for each algorithm
gridcvs = {}
inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1)

# for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3,
#                              param_grid4, param_grid5, param_grid6),
#                             (pipe1, pipe2, clf3, pipe4, clf5, pipe6),
#                             ('Softmax', 'KNN', 'DTree', 'SVM', 'RForest', 'ExtraTrees')):
#     gcv = GridSearchCV(estimator=est,
#                        param_grid=pgrid,
#                        scoring='accuracy',
#                        n_jobs=-1,
#                        cv=inner_cv,
#                        verbose=0,
#                        refit=True)
#     gridcvs[name] = gcv
#

for pgrid, est, name in zip((param_grid5, param_grid6),
                            (pipe5, pipe6),
                            ('RForest', 'ExtraTrees')):
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

for name, gs_est in sorted(gridcvs.items()):
    scores_dict = cross_validate(gs_est,
                                 X=X_train,
                                 y=y_train,
                                 cv=outer_cv,
                                 return_estimator=True,
                                 n_jobs=7)

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

# %%
gcv_model_select = GridSearchCV(estimator=pipe5,
                                param_grid=param_grid5,
                                scoring='accuracy',
                                n_jobs=-1,
                                cv=inner_cv,
                                verbose=1,
                                refit=True)

gcv_model_select.fit(X_train, y_train)

# %%
best_model = gcv_model_select.best_estimator_


## We can skip the next step because we set refit=True
## so scikit-learn has already fit the model to the
## whole training set

# best_model.fit(X_train, y_train)


train_acc = accuracy_score(y_true=y_train, y_pred=best_model.predict(X_train))
test_acc = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))

print('Accuracy %.2f%% (average over k-fold CV test folds)' %
      (100 * gcv_model_select.best_score_))
print('Best Parameters: %s' % gcv_model_select.best_params_)

print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))
