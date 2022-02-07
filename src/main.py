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
from src.evaluation.evaluator import BootstrapEvaluator
from src.services.functions import *
from src.services.paths import (
    PER_MIN_AVGS_EXCL_DEBUTS
)

# hyper parameters
batch_size = 30
epochs = 100
lr_rate = 0.25
bootstrap_sample_size = 428
bootstrap_repetitions = 100
bootstrap_repeats = False
kelly_fraction = 0.05
decorrelation_ratio = 0.5

# model choices
hidden_nodes = 100
test_set_size_ratio = 0.2
validation_set_size_ratio = 0.2
dataset = PER_MIN_AVGS_EXCL_DEBUTS
net_model = OneHiddenLayer(68, hidden_nodes)
#loss_function = ProfitLoss(kelly_fraction)
loss_function = MSEDecorrelationLoss(decorrelation_ratio)
database = 'logreg'
optimizer = 'SGD'

# prepare data
df = pd.read_csv(dataset, header=0)
train_set, test_set = split_train_test(df, test_set_size_ratio)
test_odds = get_odds(test_set)
test_results = get_results(test_set)

# train net
cur_framework = Framework(train_set, validation_set_size_ratio, net_model, loss_function, database)
cur_framework.train_model(batch_size, optimizer, lr_rate, epochs)

# evaluate on out-of-sample test data
result_probabilities, predictive_accuracy = cur_framework.predict(test_set)
evaluator = BootstrapEvaluator(result_probabilities, test_odds, test_results, kelly_fraction,
                               bootstrap_sample_size, bootstrap_repetitions)
evaluator.run_simulation()
evaluator.run_sequential_simulation()
bootstrap_roi_results = evaluator.roi_results
sequential_results = evaluator.sequential_roi

# print results
print('---Hyper parameters---')
print('Batch size: ' + str(batch_size))
print('LR rate: ' + str(lr_rate))
print('Kelly fraction: ' + str(kelly_fraction))
print('Hidden nodes: ' + str(hidden_nodes))
print('Decorrelation ratio: ' + str(0.5))
print('---Bootstrap results---')
print('ROI median: ' + str(np.median(bootstrap_roi_results)))
print('Average ROI: ' + str(bootstrap_roi_results.mean()))
print('ROI standard deviation: ' + str(bootstrap_roi_results.std()))
print('Worst-case ROI: ' + str(bootstrap_roi_results.min()))
print('Best-case ROI: ' + str(bootstrap_roi_results.max()))
print('---Sequential results---')
print('ROI: ' + str(sequential_results) + '\n')
print(bootstrap_roi_results[bootstrap_roi_results < 0].size)
