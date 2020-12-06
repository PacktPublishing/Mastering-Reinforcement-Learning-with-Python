import copy
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

config = copy.deepcopy(DEFAULT_CONFIG)

config["num_workers"] = 60
config["num_gpus"] = 1
config["batch_mode"] = "complete_episodes"
config["gamma"] = 0.999
config["lr"] = 0.0001
config["use_gae"] = False
# config["lambda"] = 1

config["sgd_minibatch_size"] = 128
config["num_sgd_iter"] = 10
config["vf_loss_coeff"] = 10
config["vf_share_layers"] = True
config["entropy_coeff"] = 0
config["clip_param"] = 0.5
config["vf_clip_param"] = 1
config["grad_clip"] = 0.01
config["kl_target"] = 0.005

config["rollout_fragment_length"] = 200
config["train_batch_size"] = 12000

config["env_config"] = {
    "isDiscrete": False,
    "renders": False,
}
