import gym
from gym.spaces import Box, Dict
import numpy as np


class MountainCar(gym.Env):
    def __init__(self, env_config={}):
        self.wrapped = gym.make("MountainCar-v0")
        self.action_space = self.wrapped.action_space
        self.t = 0
        self.reward_fun = env_config.get("reward_fun")
        self.lesson = env_config.get("lesson")
        self.use_action_masking = env_config.get("use_action_masking", False)
        self.action_mask = None
        self.reset()
        if self.use_action_masking:
            self.observation_space = Dict(
                {
                    "action_mask": Box(0, 1, shape=(self.action_space.n,)),
                    "actual_obs": self.wrapped.observation_space,
                }
            )
        else:
            self.observation_space = self.wrapped.observation_space

    def _get_obs(self):
        raw_obs = np.array(self.wrapped.unwrapped.state)
        if self.use_action_masking:
            self.update_avail_actions()
            obs = {
                "action_mask": self.action_mask,
                "actual_obs": raw_obs,
            }
        else:
            obs = raw_obs
        return obs

    def reset(self):
        self.wrapped.reset()
        self.t = 0
        self.wrapped.unwrapped.state = self._get_init_conditions()
        obs = self._get_obs()
        return obs

    def _get_init_conditions(self):
        if self.lesson == 0:
            low = 0.1
            high = 0.4
            velocity = self.wrapped.np_random.uniform(
                low=0, high=self.wrapped.max_speed
            )
        elif self.lesson == 1:
            low = -0.4
            high = 0.1
            velocity = self.wrapped.np_random.uniform(
                low=0, high=self.wrapped.max_speed
            )
        elif self.lesson == 2:
            low = -0.6
            high = -0.4
            velocity = self.wrapped.np_random.uniform(
                low=-self.wrapped.max_speed, high=self.wrapped.max_speed
            )
        elif self.lesson == 3:
            low = -0.6
            high = -0.1
            velocity = self.wrapped.np_random.uniform(
                low=-self.wrapped.max_speed, high=self.wrapped.max_speed
            )
        elif self.lesson == 4 or self.lesson is None:
            low = -0.6
            high = -0.4
            velocity = 0
        else:
            raise ValueError
        obs = (self.wrapped.np_random.uniform(low=low, high=high), velocity)
        return obs

    def set_lesson(self, lesson):
        self.lesson = lesson

    def step(self, action):
        self.t += 1
        state, reward, done, info = self.wrapped.step(action)
        if self.reward_fun == "custom_reward":
            position, velocity = state
            reward += (abs(position + 0.5) ** 2) * (position > -0.5)
        obs = self._get_obs()
        if self.t >= 200:
            done = True
        return obs, reward, done, info

    def update_avail_actions(self):
        self.action_mask = np.array([1.0] * self.action_space.n)
        pos, vel = self.wrapped.unwrapped.state
        # 0: left, 1: no action, 2: right
        if (pos < -0.3) and (pos > -0.8) and (vel < 0) and (vel > -0.05):
            self.action_mask[1] = 0
            self.action_mask[2] = 0

