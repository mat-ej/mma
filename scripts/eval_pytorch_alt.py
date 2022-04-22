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


# %% tags=["injected-parameters"]
# Parameters
target = ["R_DEC", "R_KO", "R_SUB", "B_DEC", "B_KO", "B_SUB"]
pytorch_conf = {
    "model_class": "func.mytorch.NN",
    "model_conf": {"hidden_nodes": 100, "hidden_layers": 1, "dropout_rate": 0.3},
    "optimizer_class": "torch.optim.Adam",
    "optimizer_conf": {"lr": 1e-05, "weight_decay": 0.01, "amsgrad": False},
    "loss_class": "torch.nn.CrossEntropyLoss",
    "loss_conf": {},
    "batch_size": 30,
    "epochs": 100,
}
odds_cols = [
    "R_DEC_ODDS",
    "B_DEC_ODDS",
    "R_SUB_ODDS",
    "B_SUB_ODDS",
    "R_KO_ODDS",
    "B_KO_ODDS",
]
bootstrap_repetitions = 3
kelly_fraction = 0.05
upstream = {
    "split-train-test-alt": {
        "train": "/home/m/repo/mma/products/data/train_alt.csv",
        "test": "/home/m/repo/mma/products/data/test_alt.csv",
    },
    "fit-pytorch-alt": {
        "nb": "/home/m/repo/mma/products/reports/fit_pytorch_alt.ipynb",
        "model_state_dict": "/home/m/repo/mma/products/models/pytorch_state_dict_alt.pt",
        "model": "/home/m/repo/mma/products/models/pytorch_alt.pt",
    },
}
product = {"nb": "/home/m/repo/mma/products/reports/eval_pytorch_alt.ipynb"}


# %%
import sklearn
import torch
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from func.mytorch import get_xyodds, BettingDataset, get_xyodds_alt
from src.evaluation.evaluator import BootstrapEvaluator

from sklearn.metrics import log_loss

# D(P||Q) KL divergence, relative entropy of P, Q
def kl(P, Q):
    return (P * np.log(P / Q)).sum()

def st_kl(R, Q):
    return (R[R>0] * np.log(R[R>0] / Q[R>0])).sum()


test_df = pd.read_csv(upstream['split-train-test-alt']['test'])
x_test, y_test, odds_test = get_xyodds_alt(test_df, odds_cols, target)
test_dataset = BettingDataset(x_test, y_test, odds_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

model = torch.load(upstream['fit-pytorch-alt']['model'])
model.eval()
print(model)

test_probs = np.zeros(shape=(len(test_dataset),))
accuracy = 0
for x, y, _ in test_loader:
    output = model(x)
    prediction = torch.round(output)
    correct = prediction.eq(y).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    test_probs = output.detach()

pred = torch.argmax(test_probs, dim=1)
y_true = torch.argmax(y, dim=1)
acc_model = pred.eq(y_true).sum().item() / len(y_true)

# y_hat = np.round(test_probs).astype(int)


# acc_model = (y_hat == y_test).sum() / len(y_test)

print(acc_model)
# your code here...
