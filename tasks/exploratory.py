# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
upstream = None

# This is a placeholder, leave it as None
product = None
import pandas as pd
from util.features import *
from util.ml import logit_analysis, logit_analysis_reg, get_sample_weights
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV
from statsmodels.discrete.conditional_models import ConditionalLogit
from fasttrees.fasttrees import FastFrugalTreeClassifier



# %% tags=["injected-parameters"]
# Parameters
upstream = {
    "preprocess": {
        "nb": "/home/m/repo/mma/reports/preprocess.ipynb",
        "data": "/home/m/repo/mma/products/data/mma.csv",
    }
}
product = {
    "nb": "/home/m/repo/mma/reports/exploratory.ipynb",
    "data": "/home/m/repo/mma/products/data/exploratory.csv",
}


# %%

all = pd.read_csv(upstream['preprocess']['data'], parse_dates=['DATE'])

p_r = all[target].astype(bool).sum() / len(all)
p_b = (~(all[target].astype(bool))).sum() / len(all)

print(f'prior p_r={p_r:.2f} p_b={p_b:.2f}')


# %%
print("prior alt")
print(all[target_alt].sum(axis=0) / len(all))

# pd.DataFrame(columns=['baseline', ])
# %%
print(f"baseline accuracy {p_r:.2f}")

# %%
mkt = (all['R_ODDS'] < all['B_ODDS']).astype(int)
mkt_acc = acc(all[target], mkt)

# %%
logit_analysis_reg(target, age, all)

# %%
logit_analysis_reg(target, age + categoricals, all)
# %%

logit_analysis_reg(target, red + blue, all)
# %%
import statsmodels.api as sm
# %%
'''
var selection
'''
sc = StandardScaler()
features = sc.fit_transform(all[red+blue])
features = pd.DataFrame(features, columns = all[red+blue].columns)

# %%
# fc = FastFrugalTreeClassifier()
# fc.fit(features.values, all[target].values)

# %%
# model = ConditionalLogit(all[target], features)
# mod = sm.Logit(target_df, features_df)

# %%
weights = get_sample_weights(all['DATE'], 0.005)

# %%
lasso = Lasso(alpha=0.01).fit(features, all[target], sample_weight=weights)
sel = SelectFromModel(lasso, prefit= True)
sel_feat = features.columns[sel.get_support()].to_list()
#

# %
model = SGDClassifier()
parameters = {'alpha':[0.01, 0.001, 0.0001]}
cv = GridSearchCV(model, parameters, fit_params={'sample_weight': weights})
cv.fit(features, all[target].values)

#
# weights = get_sample_weights(X_train, daily_discount_rate)
# var_selector_model = var_selector_model.fit(X_train, y_train)
# var_selector = SelectFromModel(var_selector_model, prefit=True)
# X_train_new = var_selector.transform(X_train)


# plot_fea_importance(var_selector_model, X_train)

selected_vars_ = X_train.columns[var_selector.get_support()].to_list()

