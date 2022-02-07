import pandas as pd
import numpy as np
from src.evaluation.simulation import Simulation


class BootstrapResults:
    def __init__(self, repetitions, sample_size):
        self.roi_results = np.zeros(shape=(repetitions,))
        self.budget_histories = np.zeros(shape=(repetitions, sample_size))

    def add_result(self, simulation_result, simulation_number):
        self.roi_results[simulation_number] = simulation_result.roi
        self.budget_histories[simulation_number, :] = simulation_result.budget_history


class BootstrapEvaluator:
    def __init__(self, odds, results, sample_size=None, repetitions=25, repeats=True):
        """
        :param odds: (test_set_size, 2) pandas/numpy matrix
        :param results: (test_set_size,) pd/np array... 1 - red, 0 - blue
        :param sample_size: Size of one randomly selected sample
        :param repetitions: Number of repetitions of sampling
        """
        self.odds = odds
        self.results = results
        self.num_of_tests = self.results.size
        if sample_size is None:
            self.sample_size = self.num_of_tests
        else:
            self.sample_size = sample_size
        self.num_of_repetitions = repetitions
        self.roi_results = np.zeros(shape=(repetitions,), dtype=float)
        self.sequential_roi = float(0)
        self.repeats = repeats
        self.simulator = Simulation(odds, results, 1000)

    def run_simulation(self, probabilities, kelly_fraction):
        results = BootstrapResults(self.num_of_repetitions, self.sample_size)
        for i in range(self.num_of_repetitions):
            choice = np.random.choice(self.num_of_tests, self.sample_size, replace=self.repeats)
            results.add_result(self.simulator.simulate_sequence(probabilities, choice, kelly_fraction), i)
        return results

    def run_sequential_simulation(self, probabilities, kelly_fraction):
        choice = np.linspace(self.num_of_tests - 1, 0, self.num_of_tests, endpoint=True).astype(int)
        results = BootstrapResults(1, choice.size)
        results.add_result(self.simulator.simulate_sequence(probabilities, choice, kelly_fraction), 0)
        return results

    def run_simulation_simultaneous_games(self, probabilities, kelly_fraction):
        results = BootstrapResults(self.num_of_repetitions, self.sample_size)
        for i in range(self.num_of_repetitions):
            choice = np.random.choice(self.num_of_tests, self.sample_size, replace=self.repeats)
            results.add_result(self.simulator.simulate_parallel_kelly(probabilities, choice, kelly_fraction, 10), i)
            print(i)
        return results

    def run_simulation_simultaneous_games_no_bootstrap(self, probabilities, kelly_fraction):
        results = BootstrapResults(1, self.sample_size)
        results.add_result(self.simulator.simulate_parallel_kelly(probabilities, choice=None, kelly_fraction=kelly_fraction, window_size=10), 0)
        return results
