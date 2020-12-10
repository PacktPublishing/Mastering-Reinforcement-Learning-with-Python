from copy import deepcopy

import ray
from ray import tune

from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG

import numpy as np
import gym
from gym.envs.classic_control.pendulum import PendulumEnv
from ray.rllib.env.meta_env import MetaEnv

from penenv2 import PenEnv2

config = deepcopy(DEFAULT_CONFIG)


ray.init()
tune.run(
    PPOTrainer,
    #stop={"training_iteration": 500},
    config=dict(
        DEFAULT_CONFIG,
        **{
            "env": PenEnv2,
            "horizon": 200,
            "rollout_fragment_length": 200,
            #"num_envs_per_worker": 10,
            "gamma": 0.99,
            "lambda": 1.0,
            "lr": 0.001,
            "vf_loss_coeff": 0.5,
            "clip_param": 0.3,
            "kl_target": 0.01,
            "kl_coeff": 0.001,
            "num_workers": 60,
            "num_gpus": 1,

            "clip_actions": False,
            "model": {#"fcnet_hiddens": [64, 64],
                        "use_lstm": True,
                        "lstm_cell_size": 128,
                        "lstm_use_prev_action_reward": True,
                        "max_seq_len": 10
                      }
        }
    ),
    checkpoint_freq=10,
)
