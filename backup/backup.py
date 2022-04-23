# + tags=["parameters"]
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from func.mytorch import get_xyodds, BettingDataset
from src.evaluation.evaluator import BootstrapEvaluator

upstream = None
product = None
pytorch_config = None
target = None
odds_cols = None

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
pytorch_config = {}
odds_cols = ["R_ODDS", "B_ODDS"]
upstream = {"split-train-test": {"train": "/home/m/repo/mma/products/data/train.csv", "test": "/home/m/repo/mma/products/data/test.csv"}, "fit-pytorch": {"nb": "/home/m/repo/mma/products/reports/fit_pytorch.ipynb", "model_state_dict": "/home/m/repo/mma/products/models/pytorch_state_dict.pt", "model": "/home/m/repo/mma/products/models/pytorch.pt"}}
product = {"nb": "/home/m/repo/mma/products/reports/evaluate_pytorch.ipynb"}


bootstrap_repetitions = 1
kelly_fraction = 0.05

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


