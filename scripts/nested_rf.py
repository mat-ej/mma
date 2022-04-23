# + tags=["parameters"]


upstream = None
product = None
target = None
random_seed = None
validation_ratio = None
odds_cols = None

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
random_seed = 1
validation_ratio = 0.2
odds_cols = ["R_ODDS", "B_ODDS"]
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    }
}
product = {
    "nb": "/home/m/repo/mma/products/reports/fit_pytorch.ipynb",
    "model_state_dict": "/home/m/repo/mma/products/models/pytorch_state_dict.pt",
    "model": "/home/m/repo/mma/products/models/pytorch.pt",
}
# -

# %%
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# %%
train_df = pd.read_csv(upstream['split-train-test']['train'])
test_df = pd.read_csv(upstream['split-train-test']['test'])

# %%
## Should only include fundamental vars
y = train_df[target]
X_fund = train_df.drop(columns = target)

# %%
if odds_cols in train_df.columns.tolist():
    X_fund = train_df.drop(columns=odds_cols)


# X = X_all.drop(columns = odds_cols)
print(X_fund.columns)


# %%
# n = int(train_df.shape[0] * validation_ratio)
# train_df = train_df.iloc[n:]
# val_df = train_df.iloc[0:n]
print(X_fund.head())
print(y.head())

X = X_fund.values
y = y.values.ravel()

# %%
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
model = RandomForestClassifier(random_state=random_seed)
space = dict()
space['n_estimators'] = [50, 100, 500, 1000, 10000]
space['max_features'] = list(range(10, len(X_fund.columns) + 1, 10))

# %%
# nested CV
cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_seed)
cv_inner = KFold(n_splits=2, shuffle=True, random_state=random_seed)

# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
# execute the nested cross-validation
scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
# report performance
print(scores)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
print(search.best_estimator_)
# outer_results = list()
# for train_ix, test_ix in cv_outer.split(X):
# 	# split data
# 	X_train, X_test = X[train_ix, :], X[test_ix, :]
# 	y_train, y_test = y[train_ix], y[test_ix]
# 	# configure the cross-validation procedure
# 	cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_seed)
# 	# define the model
# 	model = RandomForestClassifier(random_state=1)
# 	# define search space
# 	space = dict()
# 	space['n_estimators'] = [10, 100, 500, 1000]
# 	space['max_features'] = [2, 4, 6, 8, 10, 12]
# 	# define search
# 	search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
# 	# execute search
# 	result = search.fit(X_train, y_train)
# 	# get the best performing model fit on the whole training set
# 	best_model = result.best_estimator_
# 	# evaluate model on the hold out dataset
# 	yhat = best_model.predict(X_test)
# 	# evaluate the model
# 	acc = accuracy_score(y_test, yhat)
# 	# store the result
# 	outer_results.append(acc)
# 	# report progress
# 	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

# print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
