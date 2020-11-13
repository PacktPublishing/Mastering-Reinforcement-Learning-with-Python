import numpy as np
import ray
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer

from inventory_env import InventoryEnv

config = DEFAULT_CONFIG.copy()
config["env"] = InventoryEnv

ray.init()
trainer = PPOTrainer(config=config, env=InventoryEnv)

trainer.restore(
    # Replace this with your checkpoint path.
    "/home/enes/ray_results/PPO_InventoryEnv_2020-10-06_04-58-04t8r36o9o/checkpoint_781/checkpoint-781"
)

if __name__ == "__main__":
    np.random.seed(0)
    env = InventoryEnv()
    episode_reward_avgs = []
    episode_total_rewards = []
    for i in range(2000):
        print(f"Episode: {i+1}")
        state = env.reset()
        done = False
        ep_rewards = []
        while not done:
            action = trainer.compute_action(state)
            state, reward, done, info = env.step(action)
            ep_rewards.append(reward)
        total_reward = np.sum(ep_rewards)
        reward_per_day = np.mean(ep_rewards)
        print(f"Total reward: {total_reward}")
        print(f"Reward per time step: {reward_per_day}")
        episode_reward_avgs.append(reward_per_day)
        episode_total_rewards.append(total_reward)
        print(
            f"Average daily reward over {len(episode_reward_avgs)} "
            f"test episodes: {np.mean(episode_reward_avgs)}. "
            f"Average total epsisode reward: {np.mean(episode_total_rewards)}"
        )