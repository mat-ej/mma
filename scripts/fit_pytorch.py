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
from torch.nn import BCELoss

from func.mytorch import BettingDataset, NN, get_xyodds
from src.evaluation.evaluator import BootstrapEvaluator

upstream = None
product = None
pytorch_config = None
target = None
validation_ratio = None

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
pytorch_config = {}
validation_ratio = 0.2
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    }
}
product = {
    "nb": "/home/m/repo/mma/products/reports/fit_pytorch.ipynb",
    "model": "/home/m/repo/mma/products/models/pytorch.pt",
}

# -

from src.pytorch.pytorch_framework import *
from src.services.functions import *
from src.services.paths import *
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# class BettingDataset(Dataset):
#     def __init__(self, x, y, odds):
#         self.x = x
#         self.y = y
#         self.odds = odds
#     def __len__(self):
#         return self.y.shape[0]
#
#     def __getitem__(self, idx):
#         return [self.x[idx], self.y[idx], self.odds[idx]]
#
#     def n_features(self):
#         return self.x.shape[1]
#
# class NN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_nodes, hidden_layers, dropout_rate):
#         super(NN, self).__init__()
#
#         self.n_hidden_layers = hidden_layers
#
#         if self.n_hidden_layers == 0:
#             self.l1 = nn.Linear(input_dim, 1)
#             nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
#         else:
#             self.l1 = nn.Linear(input_dim, hidden_nodes)
#             nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
#             self.hidden_layers = []
#             for i in range(0, hidden_layers - 1):
#                 self.hidden_layers.append(nn.Linear(hidden_nodes, hidden_nodes))
#                 nn.init.xavier_normal_(self.hidden_layers[-1].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
#
#             self.hidden_layers.append(nn.Linear(hidden_nodes, 1))
#             nn.init.xavier_normal_(self.hidden_layers[-1].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
#
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.tanh = nn.Tanh()
#         self.leaky = nn.LeakyReLU(0.2)
#         self.silu = nn.SiLU()
#
#     def forward(self, x):
#         if self.n_hidden_layers == 0:
#             # x = self.silu(x)
#             x = self.l1(x)
#             # x = self.dropout(x)
#             x = self.sigmoid(x)
#         else:
#             #x = self.silu(x)
#             x = self.l1(x)
#             x = self.silu(x)
#             x = self.dropout(x)
#             for l in self.hidden_layers:
#                 x = l(x)
#                 x = self.dropout(x)
#                 #x = self.relu(x)
#
#             x = self.sigmoid(x)
#
#         return x


def train_model(train_dataset, validation_dataset, model, loss_function,
                batch_size, optimizer_name, lr_rate, epochs, momentum,
                filename, validation_evaluator, kelly_fraction, print_validations=True):

    device = torch.device('cpu')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset),
                                              shuffle=False)

    optimizer = get_optimizer(optimizer_name, model, lr_rate, momentum)

    top_accuracy = 0
    top_loss = np.inf

    val_losses = np.zeros(shape=(epochs,), dtype=float)
    val_accuracies = np.zeros(shape=(epochs,), dtype=float)

    train_losses = np.zeros(shape=(epochs,), dtype=float)
    train_accuracies = np.zeros(shape=(epochs,), dtype=float)
    for epoch in range(epochs):
        ep_loss = 0
        obs_count = 0
        model.train()
        for i, (x, y, odds) in enumerate(train_loader):
            x, y, odds = x.to(device), y.to(device), odds.to(device)
            model = model.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            if type(loss_function).__name__ == 'MSEDecorrelationLoss' \
                    or type(loss_function).__name__ == 'ProfitLoss'\
                    or type(loss_function).__name__ == 'JSDecorrelationLoss'\
                    or type(loss_function).__name__ == 'KLDecorrelationLoss'\
                    or type(loss_function).__name__ == 'PearsonDecorrelationLoss'\
                    or type(loss_function).__name__ == 'BCEDecorrelationLoss':
                loss = loss_function(outputs, y, odds)
            else:
                loss = loss_function(outputs, y)
            ep_loss += loss.item()
            obs_count += len(y)
            loss.backward()
            optimizer.step()

        # train_losses[epoch] = ep_loss / obs_count

        # validation
        model.eval()
        correct = 0
        loss = np.inf
        with torch.no_grad():
            for x, y, odds in validation_loader:
                x, y, odds = x.to(device), y.to(device), odds.to(device)
                model = model.to(device)
                output = model(x)
                if epoch % 1000 == 0:
                    validation_betting_results = validation_evaluator.run_simulation(output, kelly_fraction)
                    # median = np.median(validation_betting_results.roi_results)
                    bootstrap_roi_results = validation_betting_results.roi_results
                    print('ROI median: ' + str(np.median(bootstrap_roi_results)))
                    print('Average ROI: ' + str(bootstrap_roi_results.mean()))
                    print('ROI standard deviation: ' + str(bootstrap_roi_results.std()))
                    print('Worst-case ROI: ' + str(bootstrap_roi_results.min()))
                    print('Best-case ROI: ' + str(bootstrap_roi_results.max()))
                    print('Percentage of profitable simulations: ' + str(
                        bootstrap_roi_results[bootstrap_roi_results >= 0].size /
                        bootstrap_roi_results.size * 100) + '%')

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

        val_accuracy = correct / len(validation_loader.dataset)
        val_losses[epoch] = loss
        val_accuracies[epoch] = val_accuracy
        if print_validations:
            # print('[VAL] Validation accuracy: {:.2f}%'.format(100 * val_accuracy))
            # print('[VAL] Validation loss: {:.4f}'.format(loss))
            print('[epoch = {}] Validation loss: {:.4f} Validation accuracy: {:.2f}%'.format(epoch + 1, loss, 100 * val_accuracy))

        if val_accuracy > top_accuracy:
            top_accuracy = val_accuracy

        if loss < top_loss:
            top_loss = loss
            torch.save(model.state_dict(), filename)

    #TODO running loss
    return val_losses, val_accuracies


train_df = pd.read_csv(upstream['split-train-test']['train'])
test_df = pd.read_csv(upstream['split-train-test']['test'])
n = int(train_df.shape[0] * validation_ratio)
train_df = train_df.iloc[n:]
val_df = train_df.iloc[0:n]

print(train_df.columns)

x = train_df.drop(columns = target).astype(np.float32).values
y = train_df[target].astype(np.float32).values.reshape(-1, 1)
scaler = StandardScaler()
x = scaler.fit_transform(x)
odds = train_df[['R_ODDS', 'B_ODDS']].astype(np.float32).values

odds_cols = ['R_ODDS', 'B_ODDS']
x_train, y_train, odds_train = get_xyodds(train_df, odds_cols, target)
x_val, y_val, odds_val = get_xyodds(val_df, odds_cols, target)
x_test, y_test, odds_test = get_xyodds(test_df, odds_cols, target)

train_dataset = BettingDataset(x_train, y_train, odds_train)
validation_dataset = BettingDataset(x_val, y_val, odds_val)
test_dataset = BettingDataset(x_test, y_test, odds_test)

# hyper parameters
batch_size = 80
epochs = 100
lr_rate = 0.00008
decorrelation_ratio = 0.4
momentum = 0.9

# model choices
hidden_nodes = 100
dropout_rate = 0.25
validation_set_size_ratio = 0.25
kelly_fraction = 0.05

dataset = PER_MIN_WEIGHTED_NO_DEBUTS


net_model = OneHiddenLayer(58, hidden_nodes, dropout_rate)
model = NN(input_dim=58, hidden_nodes=100, hidden_layers=1, dropout_rate=dropout_rate)
# loss_function = BCEDecorrelationLoss(decorrelation_ratio)
loss_function = BCELoss()
optimizer_name = 'SGD'
# dataset_name = 'basic'
#
# # filepath to save model at
# current_file = os.path.abspath(os.path.dirname(__file__))
filename = product['model']
#
# # prepare data
# df = pd.read_csv(dataset, header=0)
# train_set, _ = split_train_test(df, test_set_size_ratio)
validation_evaluator = BootstrapEvaluator(odds=odds_val, results=y_val, sample_size=len(y_val), repetitions=100)


pytorch_config = {
    'train_dataset': train_dataset,
    'validation_dataset': validation_dataset,
    'model': model,
    'loss_function':loss_function,
    'batch_size':batch_size,
    'optimizer_name':optimizer_name,
    'lr_rate':lr_rate,
    'epochs':epochs,
    'momentum':momentum,
    'filename':filename,
    'validation_evaluator':validation_evaluator,
    'kelly_fraction':kelly_fraction,
    'print_validations': True
}
# train net
val_losses, val_accuracies = train_model(**pytorch_config)


# loss_val = val_losses
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, val_losses, 'b', label='validation loss')
plt.plot(epochs_range, val_accuracies, 'r', label='validation accuracy')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.load_state_dict(torch.load(product['model']))
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

bootstrap_repetitions = 10
kelly_fraction = 0.05

model.eval()
test_probs = np.zeros(shape=(len(test_dataset),))
accuracy = 0
for x, y, _ in test_loader:
    output = model(x)
    prediction = torch.round(output)
    correct = prediction.eq(y).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    test_probs = output.detach().numpy().reshape(-1, )


evaluator = BootstrapEvaluator(odds_test, y_test, len(y_test), bootstrap_repetitions)
simultaneous_results = evaluator.run_simulation_simultaneous_games(test_probs, kelly_fraction)

bootstrap_roi_results = simultaneous_results.roi_results
seq_results = evaluator.run_simulation_simultaneous_games_no_bootstrap(test_probs, kelly_fraction)

# print results
print('---Predictive accuracy---')
print('Accuracy: ' + str(accuracy) + '%')
print('---Bootstrap results---')
print('ROI median: ' + str(np.median(bootstrap_roi_results)))
print('Average ROI: ' + str(bootstrap_roi_results.mean()))
print('ROI standard deviation: ' + str(bootstrap_roi_results.std()))
# print('Worst-case ROI: ' + str(bootstrap_roi_results.min()))
# print('Best-case ROI: ' + str(bootstrap_roi_results.max()))
print('Percentage of profitable simulations: ' + str(bootstrap_roi_results[bootstrap_roi_results >= 1.0].size /
                                                     bootstrap_roi_results.size * 100) + '%')
print('---Sequential results---')

market_predictions = (odds_test[:,0] <= odds_test[:,1]).astype(int)
test_results = y_test.flatten().astype(int)
market_accuracy = np.sum(market_predictions == test_results) / len(test_results)

print(market_accuracy)
np.sum(market_predictions == test_results) / len(test_results)
print(seq_results.roi_results)















# loss_function = MSEDecorrelationLoss(decorrelation_ratio)
# loss_function = JSDecorrelationLoss(decorrelation_ratio)
# loss_function = KLDecorrelationLoss(decorrelation_ratio)
