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
from util.paths import *
from util.preproc import *
from util.features import *

# %% tags=["injected-parameters"]
# Parameters
product = {
    "nb": "/home/m/repo/mma/reports/preprocess.ipynb",
    "data": "/home/m/repo/mma/products/data/mma.csv",
}


# %%
all = load_data()
all['R_DEC'] = all[decisions].any(axis=1).astype(int) * all[target].astype(bool)
all['B_DEC'] = all[decisions].any(axis=1).astype(int) * ~all[target].astype(bool)
all['R_KO'] = all['KO/TKO'] * all[target].astype(bool)
all['B_KO'] = all['KO/TKO'] * ~all[target].astype(bool)
all['R_SUB'] = all['SUBMISSION'] * all[target].astype(bool)
all['B_SUB'] = all['SUBMISSION'] * ~all[target].astype(bool)

# %%
all.rename(columns={
 'CATCH WEIGHT': 'CATCH_WEIGHT',
 'LIGHT HEAVYWEIGHT': 'LIGHT_HEAVYWEIGHT',
 "WOMEN'S BANTAMWEIGHT": "WOMENS_BANTAMWEIGHT",
 "WOMEN'S FEATHERWEIGHT": "WOMENS_FEATHERWEIGHT",
 "WOMEN'S FLYWEIGHT": "WOMENS_FLYWEIGHT",
 "WOMEN'S STRAWWEIGHT": "WOMENS_STRAWWEIGHT",
 "KO/TKO": "KO_TKO"
},
inplace=True)

# %%
assert len(all) == all[target_alt].sum(axis=1).sum(), 'The length and alt target do not match'

# %%
print("debuts should be zero for all")

print(all[debuts].sum())

all.to_csv(product['data'], index = False)

# your code here...
