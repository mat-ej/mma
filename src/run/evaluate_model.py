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
import scipy.stats as st
from src.services.paths import *

# hyper parameters
bootstrap_sample_size = 536
bootstrap_repetitions = 1000
bootstrap_repeats = False
kelly_fraction = 0.1

# model choices
hidden_nodes = 100
dropout_rate = 0.25

test_set_size_ratio = 0.2
dataset = PER_MIN_WEIGHTED_NO_DEBUTS
net_model = OneHiddenLayer(58, hidden_nodes, dropout_rate)
dataset_name = 'basic'

# filepath to load model from
current_file = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(current_file, '/home/m/repo/mma/final_models/MSE.pt')


# prepare data
df = pd.read_csv(dataset, header=0)
_, test_set = split_train_test(df, test_set_size_ratio)
test_odds = get_odds(test_set)
test_results = get_results(test_set)

# evaluate on out-of-sample test data
result_probabilities, predictive_accuracy = predict(test_set, net_model, dataset_name, filename)
evaluator = BootstrapEvaluator(test_odds, test_results, bootstrap_sample_size, bootstrap_repetitions)

bootstrap_results = evaluator.run_simulation(result_probabilities, kelly_fraction)
bootstrap_roi_results = bootstrap_results.roi_results
sequential_results = evaluator.run_sequential_simulation(result_probabilities, kelly_fraction)

# print results
print('---Predictive accuracy---')
print('Accuracy: ' + str(predictive_accuracy) + '%')
print('---Bootstrap results---')
print('ROI median: ' + str(np.median(bootstrap_roi_results)))
print('Average ROI: ' + str(bootstrap_roi_results.mean()))
print('ROI standard deviation: ' + str(bootstrap_roi_results.std()))
print('Worst-case ROI: ' + str(bootstrap_roi_results.min()))
print('Best-case ROI: ' + str(bootstrap_roi_results.max()))
print('Percentage of profitable simulations: ' + str(bootstrap_roi_results[bootstrap_roi_results >= 0].size /
                                                     bootstrap_roi_results.size * 100) + '%')
print('---Sequential results---')
print('ROI: ' + str(sequential_results.roi_results[0]))

