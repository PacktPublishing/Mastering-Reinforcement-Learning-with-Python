"""
Mostly copied from the test scripts in
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/
"""
import time
import gym
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from custom_kuka import CustomKukaEnv
from configs import config


def main(checkpoint):
    env = CustomKukaEnv(dict(renders=True, isDiscrete=False, maxSteps=10000000))

    class EnvPlaceholder(gym.Env):
        def __init__(self, env_config):
            super(EnvPlaceholder, self).__init__()
            self.observation_space = env.observation_space
            self.action_space = env.action_space

    trainer = PPOTrainer(config=config, env=EnvPlaceholder)

    trainer.restore(checkpoint)
    done = False
    i = 0
    while not done:
        time.sleep(0.01)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        obs = env.getExtendedObservation()
        print(i)
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        i += 1


if __name__ == "__main__":
    checkpoint = "/home/enes/ray_results/PPO/PPO_ALPKukaEnv_91ddd_00000_0_2020-11-01_00-12-15/checkpoint_140/checkpoint-140"
    ray.init()
    main(checkpoint)
