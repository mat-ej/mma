import torch
from src.pytorch.pytorch_datasets import *
from src.pytorch.pytorch_loss_functions import *
from src.pytorch.pytorch_net_models import *
import torch.nn as nn


def train_model(training_set, validation_ratio, model, loss_function, dataset_name, batch_size, optimizer_name, lr_rate,
                epochs, momentum, filename, print_validations=True):
    # set up data and environment
    n = int(training_set.shape[0] * validation_ratio)
    train_set = training_set.iloc[n:]
    test_set = training_set.iloc[0:n]
    train_dataset = get_dataset(train_set, dataset_name)
    validation_dataset = get_dataset(test_set, dataset_name)

    # device = torch.device('cuda')
    device = torch.device('cpu')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset),
                                              shuffle=False)

    optimizer = get_optimizer(optimizer_name, model, lr_rate, momentum)

    top_accuracy = 0
    top_loss = np.inf

    val_losses = np.zeros(shape=(epochs,), dtype=float)
    accuracies = np.zeros(shape=(epochs,), dtype=float)
    for epoch in range(epochs):
        ep_loss = 0
        model.train()
        for i, (x, y, odds) in enumerate(train_loader):
            x, y, odds = x.to(device), y.to(device), odds.to(device)
            model = model.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            if type(loss_function).__name__ == 'MSEDecorrelationLoss' or type(loss_function).__name__ == 'ProfitLoss'\
                    or type(loss_function).__name__ == 'JSDecorrelationLoss'\
                    or type(loss_function).__name__ == 'KLDecorrelationLoss'\
                    or type(loss_function).__name__ == 'PearsonDecorrelationLoss'\
                    or type(loss_function).__name__ == 'BCEDecorrelationLoss':
                loss = loss_function(outputs, y, odds)
            else:
                loss = loss_function(outputs, y)
            ep_loss += loss.item()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        loss = np.inf
        with torch.no_grad():
            for x, y, odds in test_loader:
                x, y, odds = x.to(device), y.to(device), odds.to(device)
                model = model.to(device)
                output = model(x)
                if type(loss_function).__name__ == 'MSEDecorrelationLoss'\
                        or type(loss_function).__name__ == 'ProfitLoss' \
                        or type(loss_function).__name__ == 'JSDecorrelationLoss' \
                        or type(loss_function).__name__ == 'KLDecorrelationLoss'\
                        or type(loss_function).__name__ == 'PearsonDecorrelationLoss'\
                        or type(loss_function).__name__ == 'BCEDecorrelationLoss':
                    loss = loss_function(output, y, odds)
                else:
                    loss = loss_function(output, y)
                prediction = torch.round(output)
                correct += prediction.eq(y).sum().item()

        val_accuracy = correct / len(test_loader.dataset)
        val_losses[epoch] = loss
        accuracies[epoch] = val_accuracy
        if print_validations:
            print('[VAL] Validation accuracy: {:.2f}%'.format(100 * val_accuracy))
            print('[VAL] Validation loss: {:.4f}'.format(loss))

        if val_accuracy > top_accuracy:
            top_accuracy = val_accuracy

        if loss < top_loss:
            top_loss = loss
            torch.save(model.state_dict(), filename)
    return val_losses, accuracies


def predict(test_data, model, dataset_name, state_dict_filename=None):
    test_dataset = get_dataset(test_data, dataset_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    if state_dict_filename is not None:
        model.load_state_dict(torch.load(state_dict_filename))

    model.eval()
    probabilities = np.zeros(shape=(len(test_dataset),))
    accuracy = 0
    for x, y, _ in test_loader:
        output = model(x)
        prediction = torch.round(output)
        correct = prediction.eq(y).sum().item()
        accuracy = 100 * correct / len(test_loader.dataset)
        probabilities = output.detach().numpy().reshape(-1, )
    return probabilities, accuracy


def get_optimizer(optimizer_name, model, lr_rate, momentum):
    if optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=momentum)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr_rate)


def get_dataset(df, dataset_name):
    if dataset_name == 'basic':
        return BasicDataset(df)
    elif dataset_name == 'filtered':
        return FilteredDataset(df)
    elif dataset_name == 'weighted':
        return WeightedDataset(df)
    elif dataset_name == 'comparison':
        return ComparisonDataset(df)
    elif dataset_name == 'no_odds':
        return NoOddsDataset(df)

