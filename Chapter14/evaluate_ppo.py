import copy
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from custom_kuka import CustomKukaEnv, ALPKukaEnv
from configs import config


agents = {
    "ALP-GMM": "/home/enes/ray_results/PPO/PPO_ALPKukaEnv_91ddd_00000_0_2020-11-01_00-12-15/checkpoint_140/checkpoint-140",
    "Manual": "/home/enes/ray_results/PPO/PPO_CustomKukaEnv_9829b_00000_0_2020-11-01_00-12-25/checkpoint_140/checkpoint-140",
    "No Curriculum": "/home/enes/ray_results/PPO/PPO_CustomKukaEnv_9e28e_00000_0_2020-11-01_00-12-35/checkpoint_140/checkpoint-140",
}
envs = {"ALP-GMM": ALPKukaEnv, "Manual": CustomKukaEnv, "No Curriculum": CustomKukaEnv}

ray.init()
results = {}
N = 100
config["num_workers"] = 1
config["num_gpus"] = 0

# You may have to run each agents in separate sessions
# to avoid PyBullet restrictions
agent = "ALP-GMM"
# agent = "Manual"
# agent = "No Curriculum"

print(f"Evaluating agent: {agent}")
results[agent] = []
trainer = PPOTrainer(config=config, env=envs[agent])
trainer.restore(agents[agent])
env = envs[agent](dict(config["env_config"], **{"in_training": False}))
for i in range(N):
    print(agent, i)
    done = False
    obs = env.reset()
    ep_reward = 0
    while not done:
        action = trainer.compute_action(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        if done:
            obs = env.reset()
            results[agent].append(ep_reward)
print(f"Agent {agent} score: {np.round(np.mean(results[agent]), 2)}")
