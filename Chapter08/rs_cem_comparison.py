from math import sin, pi
import random
import numpy as np
from scipy.stats import norm

def black_box_projectile(theta, v0=10, g=9.81):
    assert theta >= 0
    assert theta <= 90
    return (v0 ** 2) * sin(2 * pi * theta / 180) / g


def random_shooting(n=1, min_a=0, max_a=90):
    assert min_a <= max_a
    return [random.uniform(min_a, max_a) for i in range(n)]


def pick_elites(actions, M_elites):
    actions = np.array(actions)
    assert M_elites <= len(actions)
    assert M_elites > 0
    results = np.array([black_box_projectile(a)
                        for a in actions])
    sorted_ix = np.argsort(results)[-M_elites:][::-1]
    return actions[sorted_ix], results[sorted_ix]

def try_random_shooting(trials=10000, n=20):
    print("Trying random shooting.")
    best_results = []
    best_actions = []
    for i in range(trials):
        actions_to_try = random_shooting(n)
        best_action, best_result = pick_elites(actions_to_try, 1)
        best_results.append(best_result[0])
        best_actions.append(best_action[0])
    print(f"Out of {trials} trials:")
    print(f"- Average score is {round(np.mean(best_results), 2)}")
    print(f"- Average action is {round(np.mean(best_actions), 2)}")
    print(f"- Action SD is {round(np.std(best_actions), 2)}")


def try_cem_normal(trials=10000, N=5, M_elites=2, iterations=3):
    print("Trying CEM.")
    best_results = []
    best_actions = []
    for i in range(trials):
        print(f"================ trial {i}")
        print("--iteration: 1")
        actions_to_try = random_shooting(N)
        elite_acts, _ = pick_elites(actions_to_try, M_elites)
        print(f"actions_to_try: {np.round(actions_to_try, 2)}")
        print(f"elites: {np.round(elite_acts, 2)}")
        for r in range(iterations - 1):
            print(f"--iteration: {r + 2}")
            mu, std = norm.fit(elite_acts)
            print(f"fitted normal mu: {np.round(mu, 2)}, std: {np.round(std, 2)}")
            actions_to_try = np.clip(norm.rvs(mu, std, N), 0, 90)
            elite_acts, elite_results = pick_elites(actions_to_try,
                                                    M_elites)
            print(f"actions_to_try: {np.round(actions_to_try, 2)}")
            print(f"elites: {np.round(elite_acts, 2)}")
        mu, std = norm.fit(elite_acts)
        print(f"final action: {np.round(mu, 2)}")
        best_results.append(black_box_projectile(mu))
        best_actions.append(np.clip(mu, 0, 90))
    print(f"Out of {trials} trials:")
    print(f"- Average score is {round(np.mean(best_results), 2)}")
    print(f"- Average action is {round(np.mean(best_actions), 2)}")
    print(f"- Action SD is {round(np.std(best_actions), 2)}")