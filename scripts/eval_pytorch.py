# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# + tags=["parameters"]
import sklearn
import torch
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from func.mytorch import get_xyodds, BettingDataset
from src.evaluation.evaluator import BootstrapEvaluator

from sklearn.metrics import log_loss

upstream = None
product = None
pytorch_conf = None
target = None
odds_cols = None
bootstrap_repetitions = None
kelly_fraction = None

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
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
odds_cols = ["R_ODDS", "B_ODDS"]
bootstrap_repetitions = 3
kelly_fraction = 0.05
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    },
    "fit-pytorch": {
        "nb": "/home/m/repo/mma/products/reports/fit_pytorch.ipynb",
        "model_state_dict": "/home/m/repo/mma/products/models/pytorch_state_dict.pt",
        "model": "/home/m/repo/mma/products/models/pytorch.pt",
    },
}
product = {"nb": "/home/m/repo/mma/products/reports/eval_pytorch.ipynb"}
# -

# +
# D(P||Q) KL divergence, relative entropy of P, Q
def kl(P, Q):
    return (P * np.log(P / Q)).sum()

def st_kl(R, Q):
    return (R[R>0] * np.log(R[R>0] / Q[R>0])).sum()


test_df = pd.read_csv(upstream['split-train-test']['test'])
x_test, y_test, odds_test = get_xyodds(test_df, odds_cols, target)
test_dataset = BettingDataset(x_test, y_test, odds_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

model = torch.load(upstream['fit-pytorch']['model'])
model.eval()
print(model)

test_probs = np.zeros(shape=(len(test_dataset),))
accuracy = 0
for x, y, _ in test_loader:
    output = model(x)
    prediction = torch.round(output)
    correct = prediction.eq(y).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    test_probs = output.detach().numpy()

y_hat = np.round(test_probs).astype(int)


acc_model = (y_hat == y_test).sum() / len(y_test)

print(acc_model)

# -
# y_book_hat = (odds_test[:,0] < odds_test[:,1]).astype(int).reshape(-1,1)
# acc_book = (y_book_hat == y_test).sum() / len(y_test)
#
# print("Test ACC(R,PRED): book={:.3f} model={:.3f}".format(acc_book, acc_model))
#
# p_m = np.hstack([test_probs, 1 - test_probs])
# p_b = sklearn.preprocessing.normalize(1 / odds_test, norm="l1")
# p_r = np.hstack([y_test, 1 - y_test])
#
# kl_model = st_kl(p_r, p_m) / len(y_test)
# kl_book = st_kl(p_r, p_b) / len(y_test)
#
# ll_model = log_loss(p_r, p_m)
# ll_book = log_loss(p_r, p_b)
#
# jsd_model = jensenshannon(p_r, p_m, axis=1).sum() / len(y_test)
# jsd_book = jensenshannon(p_r, p_b, axis=1).sum() / len(y_test)
#
# # metrics.log_loss(data_Y, predicted)
#
# print("Test LL(R, PRED): book={:.3f} model={:.3f}".format(ll_book, ll_model))
#
# print("Test KL(R, PRED): book={:.3f} model={:.3f}".format(kl_book, kl_model))
#
# print("Test JSD(R,PRED): book={:.3f} model={:.3f}".format(jsd_book, jsd_model))
#
# jsd_decor = np.nan_to_num(jensenshannon(p_m, p_b, axis=1), 0.0).sum() / len(y_test)
#
# print("Test JSD(BOOK,MODEL): jsd={:.3f}".format(jsd_decor))
#



# count = len(df_all) / n
#
# avg_adv = (st_kl(P_R, P_B) - st_kl(P_R, P_M)) / count
# avg_adv_log_n = avg_adv / np.log(n)
# print("log_kl=%f" % avg_adv_log_n)
# print(avg_adv_log_n)
#
# avg_bookie_kl = st_kl(P_R, P_B) / count
#
# eff = 1 - avg_bookie_kl / np.log(n)
# print("eff=%f" % eff)
#
# sum_odds_avg = np.sum(1 / O) / count
# track_take_avg = 1 - 1 / sum_odds_avg
# print("tt=%f" % track_take_avg)
# avg_adv = (st_kl(P_R, P_B) - st_kl(P_R, P_M)) / count
# 3.71%



# evaluator = BootstrapEvaluator(odds_test, y_test, len(y_test), bootstrap_repetitions)
# simultaneous_results = evaluator.run_simulation_simultaneous_games(test_probs, kelly_fraction)
#
# bootstrap_roi_results = simultaneous_results.roi_results
# seq_results = evaluator.run_simulation_simultaneous_games_no_bootstrap(test_probs, kelly_fraction)
#
# # print results
# print('---Predictive accuracy---')
# print('Accuracy: ' + str(accuracy) + '%')
# print('---Bootstrap results---')
# print('ROI median: ' + str(np.median(bootstrap_roi_results)))
# print('Average ROI: ' + str(bootstrap_roi_results.mean()))
# print('ROI standard deviation: ' + str(bootstrap_roi_results.std()))
# # print('Worst-case ROI: ' + str(bootstrap_roi_results.min()))
# # print('Best-case ROI: ' + str(bootstrap_roi_results.max()))
# print('Percentage of profitable simulations: ' + str(bootstrap_roi_results[bootstrap_roi_results >= 1.0].size /
#                                                      bootstrap_roi_results.size * 100) + '%')
# print('---Sequential results---')
#
# market_predictions = (odds_test[:,0] <= odds_test[:,1]).astype(int)
# test_results = y_test.flatten().astype(int)
# market_accuracy = np.sum(market_predictions == test_results) / len(test_results)
#
# print(market_accuracy)
# np.sum(market_predictions == test_results) / len(test_results)
# print(seq_results.roi_results)


