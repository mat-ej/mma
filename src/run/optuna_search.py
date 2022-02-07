"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np

import optuna
from optuna.trial import TrialState

from src.evaluation.evaluator import BootstrapEvaluator
from src.pytorch.pytorch_datasets import BasicDataset
from src.pytorch.pytorch_net_models import *
from src.pytorch.pytorch_loss_functions import *
from src.services.paths import PER_MIN_WEIGHTED
from src.services.functions import *

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def get_necessities():
    df = pd.read_csv(PER_MIN_WEIGHTED, header=0)
    training_set, test_set = split_train_test(df, 0.2)
    n = int(training_set.shape[0] * 0.2)
    valid_set = training_set.iloc[0:n]
    training_set = training_set.iloc[n:]
    validation_odds = get_odds(valid_set)
    validation_results = get_results(valid_set)
    validation_evaluator_class = BootstrapEvaluator(validation_odds, validation_results, valid_set.shape[0], 100)
    return BasicDataset(training_set), BasicDataset(valid_set), validation_evaluator_class


def objective(trial):
    dropout_rate = 0.25
    loss_function_name = 'BCE'
    loss_function_name = 'MSE'
    loss_function_name = 'KL'
    loss_function_name = 'JS'
    optimizer_name = 'SGD'
    momentum = 0.9
    lr_rate = 0.00008
    hidden_nodes = 100
    batch_size = 80
    decorrelation_ratio = trial.suggest_float('decorrelation_ratio', 0, 4)

    net = OneHiddenLayer(58, hidden_nodes, dropout_rate)
    net.to(DEVICE)

    loss_function = None
    if loss_function_name == 'BCE':
        loss_function = BCEDecorrelationLoss(decorrelation_ratio)
    elif loss_function_name == 'MSE':
        loss_function = MSEDecorrelationLoss(decorrelation_ratio)
    elif loss_function_name == 'JS':
        loss_function = JSDecorrelationLoss(decorrelation_ratio, device=DEVICE)
    elif loss_function_name == 'KL':
        loss_function = KLDecorrelationLoss(decorrelation_ratio)

    optimizer = None
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr_rate, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr_rate)

    train_dataset, validation_dataset, validation_evaluator = get_necessities()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset),
                                                    shuffle=False)

    kelly_fraction = 0.05
    median = None
    for epoch in range(502):
        net.train()
        for i, (x, y, odds) in enumerate(train_loader):
            x, y, odds = x.to(DEVICE), y.to(DEVICE), odds.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(x)
            loss = loss_function(outputs, y, odds)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            # validation
            net.eval()
            with torch.no_grad():
                for x, y, odds in validation_loader:
                    x, y, odds = x.to(DEVICE), y.to(DEVICE), odds.to(DEVICE)
                    output = net(x)
                    validation_betting_results = validation_evaluator.run_simulation(output, kelly_fraction)
            median = np.median(validation_betting_results.roi_results)
            trial.report(median, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    print("one trial")
    return median


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=100, timeout=1000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
