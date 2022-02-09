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

from src.pytorch.pytorch_framework import *
from src.services.functions import *
from src.services.paths import *
import torch.nn as nn

# hyper parameters
batch_size = 80
epochs = 10001
lr_rate = 0.00008
decorrelation_ratio = 0.4
momentum = 0.9
# model choices
hidden_nodes = 100
dropout_rate = 0.25
test_set_size_ratio = 0.2
validation_set_size_ratio = 0.2
dataset = PER_MIN_WEIGHTED_NO_DEBUTS
net_model = OneHiddenLayer(58, hidden_nodes, dropout_rate)
loss_function = BCEDecorrelationLoss(decorrelation_ratio)
#loss_function = MSEDecorrelationLoss(decorrelation_ratio)
#loss_function = JSDecorrelationLoss(decorrelation_ratio)
#loss_function = KLDecorrelationLoss(decorrelation_ratio)
optimizer = 'SGD'
dataset_name = 'basic'

# filepath to save model at
current_file = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(current_file, '/home/m/repo/mma/final_models/BCE.pt')

# prepare data
df = pd.read_csv(dataset, header=0)
train_set, _ = split_train_test(df, test_set_size_ratio)

# train net
train_model(train_set, validation_set_size_ratio, net_model,
            loss_function, dataset_name, batch_size, optimizer,
            lr_rate, epochs, momentum, filename)


