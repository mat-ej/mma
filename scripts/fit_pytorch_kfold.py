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

'''
Select data
Recognize number of variables
Split training and test data
    Extract odds and results from test data
Select framework - model, criterion and optimizer
Train model
Get probabilities on test data
Select evaluator
Run evaluator
Save model's results (predictive accuracy and ROI values)
'''

# + tags=["parameters"]
from torch.utils.data import SubsetRandomSampler

upstream = None
product = None
pytorch_conf = None
target = None
random_seed = None
validation_ratio = None
odds_cols = None
k_fold = 5

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
pytorch_conf = {
    "model_class": "func.mytorch.NN",
    "model_conf": {"hidden_nodes": 60, "hidden_layers": 1, "dropout_rate": 0.3},
    "optimizer_class": "torch.optim.Adam",
    "optimizer_conf": {"lr": 1e-5, "weight_decay": 1e-3, "amsgrad": True},
    "loss_class": "torch.nn.BCELoss",
    "loss_conf": {},
    "batch_size": 100,
    "epochs": 1000,
}

# pytorch_conf = {
#     "model_class": "func.mytorch.NN",
#     "model_conf": {"hidden_nodes": 100, "hidden_layers": 1, "dropout_rate": 0.2},
#     "optimizer_class": "torch.optim.Adam",
#     "optimizer_conf": {"lr": 1e-5, "weight_decay": 1e-3, "amsgrad": True},
#     "loss_class": "torch.nn.BCELoss",
#     "loss_conf": {},
#     "batch_size": 100,
#     "epochs": 10000,
# }
random_seed = 1
validation_ratio = 0.2
odds_cols = ""
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
from func.mytorch import BettingDataset, NN, get_xyodds
from src.evaluation.evaluator import BootstrapEvaluator
from sklearn.model_selection import KFold

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
                loss = loss_function(output, y)
                prediction = torch.round(output)
                correct += prediction.eq(y).sum().item()

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


train_df = pd.read_csv(upstream['split-train-test']['train'])
test_df = pd.read_csv(upstream['split-train-test']['test'])


x_train, y_train, odds_train = get_xyodds(train_df, odds_cols, target)
x_test, y_test, odds_test = get_xyodds(test_df, odds_cols, target)

train_dataset = BettingDataset(x_train, y_train, odds_train)
# test_dataset = BettingDataset(x_test, y_test, odds_test)

model_class = eval(pytorch_conf['model_class'])
pytorch_conf['model_conf']['input_dim'] = train_dataset.x.shape[1]
model = model_class(**pytorch_conf['model_conf'])

pytorch_conf['optimizer_conf']['params'] = model.parameters()
optim_class = eval(pytorch_conf['optimizer_class'])
optimizer = optim_class(**pytorch_conf['optimizer_conf'])

loss_class = eval(pytorch_conf['loss_class'])
loss_function = loss_class(**pytorch_conf['loss_conf'])
# criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset=train_dataset, batch_size=pytorch_conf['batch_size'], shuffle=True)

optimize_config = {
    'model': model,
    'loss_function': loss_function,
    'optimizer': optimizer,
    'train_loader': train_loader,
    'epochs': pytorch_conf['epochs'],
    'statedict_filename': product['model_state_dict']
}

def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()

    for x, y, odds in dataloader:
        x, y, odds = x.to(device), y.to(device), odds.to(device)
        optimizer.zero_grad()
        model = model.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        prediction = torch.round(output.data)
        (prediction == y).sum().item()
        train_correct += (prediction == y).sum().item()
        train_loss += loss.item() * x.size(0)

    return train_loss,train_correct

def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for x, y, odds in dataloader:
        x, y, odds = x.to(device), y.to(device), odds.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        predictions = torch.round(output.data)

        val_correct += (predictions == y).sum().item()
        valid_loss += loss.item() * x.size(0)

    return valid_loss,val_correct


splits = KFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
foldperf = {}
batch_size = pytorch_conf['batch_size']
k = 5
epochs = pytorch_conf['epochs']

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = model_class(**pytorch_conf['model_conf'])
    model.to(device)
    pytorch_conf['optimizer_conf']['params'] = model.parameters()
    optimizer = optim_class(**pytorch_conf['optimizer_conf'])

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in range(epochs):
        train_loss, train_correct=train_epoch(model,device,train_loader,loss_function,optimizer)
        test_loss, test_correct=valid_epoch(model,device,test_loader,loss_function)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                             epochs,
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history


testl_f,tl_f,testa_f,ta_f=[],[],[],[]
k=5
for f in range(1,k+1):

     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

     ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
     testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))


print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))

diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}

for i in range(epochs):
      diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
      diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(k)]))
      diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
      diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_acc'][i] for f in range(k)]))


# Plot losses
plt.figure(figsize=(10,8))
plt.semilogy(diz_ep['train_loss_ep'], label='Train')
plt.semilogy(diz_ep['test_loss_ep'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.grid()
plt.legend()
plt.title('CNN loss')
plt.show()

# Plot accuracies
plt.figure(figsize=(10,8))
plt.semilogy(diz_ep['train_acc_ep'], label='Train')
plt.semilogy(diz_ep['test_acc_ep'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.grid()
plt.legend()
plt.title('CNN accuracy')
plt.show()

torch.save(model,'k_cross_CNN.pt')



# val_losses, val_accuracies = pytorch_optimize(**optimize_config)


# loss_val = val_losses
# epochs_range = range(1, pytorch_conf['epochs'] + 1)
# plt.plot(epochs_range, val_losses, 'b', label='validation loss')
# plt.plot(epochs_range, val_accuracies, 'r', label='validation accuracy')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# model.load_state_dict(torch.load(product['model_state_dict']))
# model.eval()

torch.save(model, product['model'])