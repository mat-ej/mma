import pandas as pd
import numpy as np
from src.evaluation.betting_strategies import *


class SimulationResults:
    def __init__(self, sample_size):
        self.roi = 0
        self.budget_history = np.zeros(shape=(sample_size,))


class Simulation:
    def __init__(self, odds, results, budget=1000):
        """
        :param odds: (test_set_size, 2) pandas/numpy matrix
        :param results: (test_set_size,) pd/np array... 1 - red, 0 - blue
        :param budget: initial budget
        """
        self.odds = odds
        self.results = results
        self.initial_budget = budget
        self.budget = budget

    def evaluate_bet(self, fraction, odds, win):
        if not win:
            self.budget -= self.budget * fraction
        else:
            self.budget += self.budget * fraction * (odds - 1)

    def simulate_sequence(self, probabilities, subset_choice, kelly_fraction):
        """
        :param kelly_fraction:
        :param probabilities: (test_set_size,) predicted probabilities
        :param subset_choice: subset of test set (with repeats)
        :return:
        """
        self.budget = self.initial_budget
        results = SimulationResults(subset_choice.size)
        step = 0
        for i in subset_choice:
            probability_red = probabilities[i]
            probability_blue = 1 - probability_red
            odds_red = self.odds[i, 0]
            odds_blue = self.odds[i, 1]
            win_red = self.results[i] == 1
            win_blue = self.results[i] == 0
            fraction_red = sequential_kelly(probability_red, odds_red, kelly_fraction)
            fraction_blue = sequential_kelly(probability_blue, odds_blue, kelly_fraction)
            if fraction_red > 0:
                self.evaluate_bet(fraction_red, odds_red, win_red)
            if fraction_blue > 0:
                self.evaluate_bet(fraction_blue, odds_blue, win_blue)
            results.budget_history[step] = self.budget
            step += 1
        results.roi = float((self.budget - self.initial_budget) / self.initial_budget)
        return results

    def shuffle(self, probabilities, choice):
        shuffled_probs = np.zeros(shape=(probabilities.size, 2))
        shuffled_odds = np.zeros(shape=(probabilities.size, 2))
        shuffled_results = np.zeros(shape=(probabilities.size,))
        for i in range(choice.size):
            shuffled_probs[i, 0] = probabilities[choice[i]]
            shuffled_probs[i, 1] = 1 - probabilities[choice[i]]
            shuffled_odds[i, :] = self.odds[choice[i], :]
            shuffled_results[i] = self.results[choice[i]]
        return shuffled_probs, shuffled_odds, shuffled_results

    def simulate_parallel_kelly(self, probabilities, choice=None, kelly_fraction=1, window_size=10):
        if choice is None:
            choice = np.linspace(probabilities.size - 1, 0, probabilities.size, endpoint=True).astype(int)
        shuffled_probs, shuffled_odds, shuffled_results = self.shuffle(probabilities, choice)
        done = False
        cur_start_index = 0
        cur_end_index = 0 + window_size
        total_size = self.results.shape[0]
        results = SimulationResults(total_size)
        self.budget = self.initial_budget

        while not done:
            window_budget = self.budget
            cur_window_size = cur_end_index - cur_start_index
            pred_probabilities = shuffled_probs[cur_start_index:cur_end_index, :]
            odds = shuffled_odds[cur_start_index:cur_end_index, :]
            b_star = simultaneous_kelly(odds, pred_probabilities)

            for i in range(cur_window_size):
                print('Results match ' + str(shuffled_results[cur_start_index + i] == self.results[choice[cur_start_index+i]]))
                # red fighter won
                if shuffled_results[cur_start_index + i] == 1:
                    self.budget += window_budget * kelly_fraction * b_star[i * 2] * (odds[i, 0] - 1)  # bets on red - payout
                    self.budget -= window_budget * kelly_fraction * b_star[(i * 2) + 1]  # bets on blue - decrease budget
                # blue fighter won
                elif shuffled_results[cur_start_index + i] == 0:
                    self.budget += window_budget * kelly_fraction * b_star[(i * 2) + 1] * (odds[i, 1] - 1)
                    self.budget -= window_budget * kelly_fraction * b_star[i * 2]
                results.budget_history[cur_start_index + i] = self.budget
            if cur_end_index == total_size:
                done = True
            else:
                cur_start_index += window_size
                cur_end_index = min(cur_end_index + window_size, total_size)
        results.roi = float((self.budget - self.initial_budget) / self.initial_budget)
        return results
