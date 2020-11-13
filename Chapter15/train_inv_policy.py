import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer

from inventory_env import InventoryEnv

config = DEFAULT_CONFIG.copy()
config["env"] = InventoryEnv
config["num_gpus"] = 1  # Set this to 0 if you don't have a GPU.
config["num_workers"] = 50  # Set this based on the number of CPUs on your machine

# Combination 1
# config["clip_param"] = 0.3
# config["entropy_coeff"] = 0
# config["grad_clip"] = 0.01
# config["kl_target"] = 0.05
# config["lr"] = 0.0001
# config["num_sgd_iter"] = 10
# config["sgd_minibatch_size"] = 128
# config["train_batch_size"] = 10000
# config["use_gae"] = True
# config["vf_clip_param"] = 10
# config["vf_loss_coeff"] = 1
# config["vf_share_layers"] = True

# Combination 2
config["clip_param"] = 0.3
config["entropy_coeff"] = 0
config["grad_clip"] = None
config["kl_target"] = 0.005
config["lr"] = 0.001
config["num_sgd_iter"] = 5
config["sgd_minibatch_size"] = 8192
config["train_batch_size"] = 20000
config["use_gae"] = True
config["vf_clip_param"] = 10
config["vf_loss_coeff"] = 1
config["vf_share_layers"] = False

# For better gradient estimates in the later stages
# of the training, increase the batch sizes.
# config["sgd_minibatch_size"] = 8192 * 4
# config["train_batch_size"] = 20000 * 10

ray.init()
trainer = PPOTrainer(config=config, env=InventoryEnv)

# Use this when you want to continue from a checkpoint.
# trainer.restore(
#   "/home/enes/ray_results/PPO_InventoryEnv_2020-10-06_04-31-2945lwn1wg/checkpoint_737/checkpoint-737"
# )



best_mean_reward = np.NINF
while True:
    result = trainer.train()
    print(pretty_print(result))
    mean_reward = result.get("episode_reward_mean", np.NINF)
    if mean_reward > best_mean_reward:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        best_mean_reward = mean_reward
