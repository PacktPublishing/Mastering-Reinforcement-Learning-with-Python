"""
This code is modified from:
https://github.com/awslabs/or-rl-benchmarks/blob/master/News%20Vendor/src/news_vendor_environment.py
"""

import gym
import numpy as np
from gym import spaces
from scipy.stats import poisson


class InventoryEnv(gym.Env):
    def __init__(self, config={}):
        self.l = config.get("lead time", 5)
        self.storage_capacity = 4000
        self.order_limit = 1000
        self.step_count = 0
        self.max_steps = 40

        self.max_value = 100.0
        self.max_holding_cost = 5.0
        self.max_loss_goodwill = 10.0
        self.max_mean = 200

        self.inv_dim = max(1, self.l)
        space_low = self.inv_dim * [0]
        space_high = self.inv_dim * [self.storage_capacity]
        space_low += 5 * [0]
        space_high += [
            self.max_value,
            self.max_value,
            self.max_holding_cost,
            self.max_loss_goodwill,
            self.max_mean,
        ]
        self.observation_space = spaces.Box(
            low=np.array(space_low),
            high=np.array(space_high),
            dtype=np.float32
        )

        # Action is between 0 and 1, representing order quantity from
        # 0 up to the order limit.
        self.action_space = spaces.Box(
            low=np.array([0]),
            high=np.array([1]),
            dtype=np.float32
        )
        self.state = None
        self.reset()

    def _normalize_obs(self):
        obs = np.array(self.state)
        obs[:self.inv_dim] = obs[:self.inv_dim] / self.order_limit
        obs[self.inv_dim] = obs[self.inv_dim] / self.max_value
        obs[self.inv_dim + 1] = obs[self.inv_dim + 1] / self.max_value
        obs[self.inv_dim + 2] = obs[self.inv_dim + 2] / self.max_holding_cost
        obs[self.inv_dim + 3] = obs[self.inv_dim + 3] / self.max_loss_goodwill
        obs[self.inv_dim + 4] = obs[self.inv_dim + 4] / self.max_mean
        return obs

    def reset(self):
        self.step_count = 0

        price = np.random.rand() * self.max_value
        cost = np.random.rand() * price
        holding_cost = np.random.rand() * min(cost, self.max_holding_cost)
        loss_goodwill = np.random.rand() * self.max_loss_goodwill
        mean_demand = np.random.rand() * self.max_mean

        self.state = np.zeros(self.inv_dim + 5)
        self.state[self.inv_dim] = price
        self.state[self.inv_dim + 1] = cost
        self.state[self.inv_dim + 2] = holding_cost
        self.state[self.inv_dim + 3] = loss_goodwill
        self.state[self.inv_dim + 4] = mean_demand

        return self._normalize_obs()

    def break_state(self):
        inv_state = self.state[: self.inv_dim]
        p = self.state[self.inv_dim]
        c = self.state[self.inv_dim + 1]
        h = self.state[self.inv_dim + 2]
        k = self.state[self.inv_dim + 3]
        mu = self.state[self.inv_dim + 4]
        return inv_state, p, c, h, k, mu

    def step(self, action):
        beginning_inv_state, p, c, h, k, mu = \
            self.break_state()
        action = np.clip(action[0], 0, 1)
        action = int(action * self.order_limit)
        done = False

        available_capacity = self.storage_capacity \
                             - np.sum(beginning_inv_state)
        assert available_capacity >= 0
        buys = min(action, available_capacity)
        # If lead time is zero, immediately
        # increase the inventory
        if self.l == 0:
            self.state[0] += buys
        on_hand = self.state[0]
        demand_realization = np.random.poisson(mu)

        # Compute Reward
        sales = min(on_hand,
                    demand_realization)
        sales_revenue = p * sales
        overage = on_hand - sales
        underage = max(0, demand_realization
                          - on_hand)
        purchase_cost = c * buys
        holding = overage * h
        penalty_lost_sale = k * underage
        reward = sales_revenue \
                 - purchase_cost \
                 - holding \
                 - penalty_lost_sale

        # Day is over. Update the inventory
        # levels for the beginning of the next day
        # In-transit inventory levels shift to left
        self.state[0] = 0
        if self.inv_dim > 1:
            self.state[: self.inv_dim - 1] \
                = self.state[1: self.inv_dim]
        self.state[0] += overage
        # Add the recently bought inventory
        # if the lead time is positive
        if self.l > 0:
            self.state[self.l - 1] = buys
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        # Normalize the reward
        reward = reward / 10000
        info = {
            "demand realization": demand_realization,
            "sales": sales,
            "underage": underage,
            "overage": overage,
        }
        return self._normalize_obs(), reward, done, info


def get_action_from_benchmark_policy(env):
    inv_state, p, c, h, k, mu = env.break_state()
    cost_of_overage = h
    cost_of_underage = p - c + k
    critical_ratio = np.clip(
        0, 1, cost_of_underage
              / (cost_of_underage + cost_of_overage)
    )
    horizon_target = int(poisson.ppf(critical_ratio,
                         (len(inv_state) + 1) * mu))
    deficit = max(0, horizon_target - np.sum(inv_state))
    buy_action = min(deficit, env.order_limit)
    return [buy_action / env.order_limit]


if __name__ == "__main__":
    np.random.seed(100)
    env = InventoryEnv()
    episode_reward_avgs = []
    episode_total_rewards = []
    for i in range(2000):
        print(f"Episode: {i+1}")
        initial_state = env.reset()
        done = False
        ep_rewards = []
        while not done:
            # action = env.action_space.sample()
            action = get_action_from_benchmark_policy(env)
            # print("Action: ", action)
            state, reward, done, info = env.step(action)
            # print("State: ", state)
            ep_rewards.append(reward)
        total_reward = np.sum(ep_rewards)
        reward_per_day = np.mean(ep_rewards)
        # print(f"Total reward: {total_reward}")
        # print(f"Reward per time step: {reward_per_day}")
        episode_reward_avgs.append(reward_per_day)
        episode_total_rewards.append(total_reward)
        print(
            f"Average daily reward over {len(episode_reward_avgs)} "
            f"test episodes: {np.mean(episode_reward_avgs)}. "
            f"Average total epsisode reward: {np.mean(episode_total_rewards)}"
        )
