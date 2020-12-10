import ray
from ray import tune
from inventory_env import InventoryEnv

ray.init()
tune.run(
    "PPO",
    stop={"timesteps_total": 1e6},
    num_samples=5,
    config={
        "env": InventoryEnv,
        "rollout_fragment_length": 40,
        "num_gpus": 1,
        "num_workers": 50,
        "lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
        "use_gae": tune.choice([True, False]),
        "train_batch_size": tune.choice([5000, 10000, 20000, 40000]),
        "sgd_minibatch_size": tune.choice([128, 1024, 4096, 8192]),
        "num_sgd_iter": tune.choice([5, 10, 30]),
        "vf_loss_coeff": tune.choice([0.1, 1, 10]),
        "vf_share_layers": tune.choice([True, False]),
        "entropy_coeff": tune.choice([0, 0.1, 1]),
        "clip_param": tune.choice([0.05, 0.1, 0.3, 0.5]),
        "vf_clip_param": tune.choice([1, 5, 10]),
        "grad_clip": tune.choice([None, 0.01, 0.1, 1]),
        "kl_target": tune.choice([0.005, 0.01, 0.05]),
    },
)
