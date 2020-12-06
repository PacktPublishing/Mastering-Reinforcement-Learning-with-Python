import ray
from ray import tune

from custom_kuka import CustomKukaEnv
from configs import config

ray.init()
tune.run(
    "PPO",
    config=dict(
        config,
        **{
            "env": CustomKukaEnv,
        }
    ),
    checkpoint_freq=10,
)
