# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
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

# %% tags=["parameters"]

# %%
# + tags=["parameters"]
upstream = None
product = None

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
random_seed = 1
validation_ratio = 0.2
odds_cols = ""
upstream = {
    "split-train-test-alt": {
        "train": "/home/m/repo/mma/products/data/train_alt.csv",
        "test": "/home/m/repo/mma/products/data/test_alt.csv",
    }
}
product = {
    "nb": "/home/m/repo/mma/products/reports/fit_pytorch_alt.ipynb",
    "model_state_dict": "/home/m/repo/mma/products/models/pytorch_state_dict_alt.pt",
    "model": "/home/m/repo/mma/products/models/pytorch_alt.pt",
}
# -

# +
from src.pytorch.pytorch_framework import *
from src.services.functions import *
from src.services.paths import *
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim.adam
from sklearn.model_selection import KFold

from torch.nn import BCELoss
import func
from func.mytorch import BettingDataset, NN, get_xyodds, get_xyodds_alt
from src.evaluation.evaluator import BootstrapEvaluator

torch.manual_seed(random_seed)


def pytorch_optimize(model, loss_function, optimizer, train_loader, val_loader, epochs, statedict_filename=None, debug_info=True, **kwargs):

    device = torch.device('cpu')

    top_accuracy = 0
    top_loss = np.inf
    val_losses = np.zeros(shape=(epochs,), dtype=float)
    val_accuracies = np.zeros(shape=(epochs,), dtype=float)

    for epoch in range(epochs):
        ep_loss = 0
        obs_count = 0
        model.train()
        for i, (x, y, odds) in enumerate(train_loader):
            x, y, odds = x.to(device), y.to(device), odds.to(device)
            model = model.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)

            # if type(loss_function).__name__ == 'MSEDecorrelationLoss' \
            #         or type(loss_function).__name__ == 'ProfitLoss'\
            #         or type(loss_function).__name__ == 'JSDecorrelationLoss'\
            #         or type(loss_function).__name__ == 'KLDecorrelationLoss'\
            #         or type(loss_function).__name__ == 'PearsonDecorrelationLoss'\
            #         or type(loss_function).__name__ == 'BCEDecorrelationLoss':
            #     loss = loss_function(outputs, y, odds)
            # else:
            #     loss = loss_function(outputs, y)
            ep_loss += loss.item()
            obs_count += len(y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        loss = np.inf
        with torch.no_grad():
            for x, y, odds in val_loader:
                x, y, odds = x.to(device), y.to(device), odds.to(device)
                model = model.to(device)
                output = model(x)
                # if epoch % 1000 == 0:
                #     validation_betting_results = validation_evaluator.run_simulation(output, kelly_fraction)
                #     # median = np.median(validation_betting_results.roi_results)
                #     bootstrap_roi_results = validation_betting_results.roi_results
                #     print('ROI median: ' + str(np.median(bootstrap_roi_results)))
                #     print('Average ROI: ' + str(bootstrap_roi_results.mean()))
                #     print('ROI standard deviation: ' + str(bootstrap_roi_results.std()))
                #     print('Worst-case ROI: ' + str(bootstrap_roi_results.min()))
                #     print('Best-case ROI: ' + str(bootstrap_roi_results.max()))
                #     print('Percentage of profitable simulations: ' + str(
                #         bootstrap_roi_results[bootstrap_roi_results >= 0].size /
                #         bootstrap_roi_results.size * 100) + '%')

                loss = loss_function(output, y)
                pred = torch.argmax(output, dim=1)
                y_true = torch.argmax(y, dim=1)
                correct += pred.eq(y_true).sum().item()

        val_accuracy = correct / len(val_loader.dataset)
        val_losses[epoch] = loss
        val_accuracies[epoch] = val_accuracy
        if debug_info:
            print('[epoch = {}] Validation loss: {:.4f} Validation accuracy: {:.2f}%'.format(epoch + 1, loss, 100 * val_accuracy))

        if val_accuracy > top_accuracy:
            top_accuracy = val_accuracy

        if loss < top_loss:
            top_loss = loss
            torch.save(model.state_dict(), statedict_filename)

    #TODO running loss
    return val_losses, val_accuracies


train_df = pd.read_csv(upstream['split-train-test-alt']['train'])
test_df = pd.read_csv(upstream['split-train-test-alt']['test'])


n = int(train_df.shape[0] * validation_ratio)
train_df = train_df.iloc[n:]
val_df = train_df.iloc[0:n]

x_train, y_train, odds_train = get_xyodds_alt(train_df, odds_cols, target)
x_val, y_val, odds_val = get_xyodds_alt(val_df, odds_cols, target)
x_test, y_test, odds_test = get_xyodds_alt(test_df, odds_cols, target)

train_dataset = BettingDataset(x_train, y_train, odds_train)
val_dataset = BettingDataset(x_val, y_val, odds_val)
test_dataset = BettingDataset(x_test, y_test, odds_test)

model_class = eval(pytorch_conf['model_class'])
pytorch_conf['model_conf']['input_dim'] = train_dataset.x.shape[1]
pytorch_conf['model_conf']['output_dim'] = train_dataset.y.shape[1]
model = model_class(**pytorch_conf['model_conf'])

pytorch_conf['optimizer_conf']['params'] = model.parameters()
optim_class = eval(pytorch_conf['optimizer_class'])
optimizer = optim_class(**pytorch_conf['optimizer_conf'])

loss_class = eval(pytorch_conf['loss_class'])
loss_function = loss_class(**pytorch_conf['loss_conf'])

train_loader = DataLoader(dataset=train_dataset, batch_size=pytorch_conf['batch_size'], shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)

optimize_config = {
    'model': model,
    'loss_function': loss_function,
    'optimizer': optimizer,
    'train_loader': train_loader,
    'val_loader': val_loader,
    'epochs': pytorch_conf['epochs'],
    'statedict_filename': product['model_state_dict']
}
val_losses, val_accuracies = pytorch_optimize(**optimize_config)

# loss_val = val_losses
epochs_range = range(1, pytorch_conf['epochs'] + 1)
plt.plot(epochs_range, val_losses, 'b', label='validation loss')
plt.plot(epochs_range, val_accuracies, 'r', label='validation accuracy')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.load_state_dict(torch.load(product['model_state_dict']))
model.eval()

torch.save(model, product['model'])
# -
