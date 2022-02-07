import numpy as np
import scipy.optimize as sco
import cvxpy as cvx
import itertools

b_eps = 1e-8
VERBOSE = False


def sequential_kelly(probability, decimal_odds, fraction):
    p = probability
    b = decimal_odds - 1
    q = 1 - p
    position = ((p * b) - q) / b
    ev = probability * decimal_odds
    if position < 0:
        return 0
    else:
        return fraction * position


def get_payout_matrix(odds):
    n = odds.shape[0]
    variations = np.array([np.array(item, dtype=int) for item in itertools.product('10', repeat=n)])
    matrix = np.zeros(shape=(variations.shape[0], (2 * n) + 1), dtype=float)
    matrix[:, -1] = 1
    for j in range(n):
        matrix[:, j * 2] = np.where(variations[:, j] == 1, odds[j, 0], 0)
        matrix[:, (j * 2) + 1] = np.where(variations[:, j] == 0, odds[j, 1], 0)
    return matrix


def get_combination_probabilities(probs):
    n = probs.shape[0]
    variations = np.array([np.array(item, dtype=int) for item in itertools.product('10', repeat=n)])
    matrix = np.ones(shape=(1, variations.shape[0]))
    for j in range(n):
        matrix[0, :] = np.where(variations[:, j] == 1, matrix * probs[j, 0], matrix * probs[j, 1])
    return matrix


def simultaneous_kelly(odds, probs):
    n = probs.shape[0]
    b = cvx.Variable((2 * n) + 1)
    R = get_payout_matrix(odds)
    p = get_combination_probabilities(probs)

    goal = cvx.Maximize(p @ cvx.log(R @ b))
    constraints = [cvx.sum(b) == 1, b >= 0]
    problem = cvx.Problem(goal, constraints)
    try:
        problem.solve(solver='ECOS', verbose=False)
        b_star = b.value
    except Exception as e:
        b_star = np.zeros((2 * n) + 1)
    return b_star

