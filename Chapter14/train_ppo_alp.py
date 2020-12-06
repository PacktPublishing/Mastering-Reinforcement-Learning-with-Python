from ray import tune
import ray

from custom_kuka import ALPKukaEnv
from configs import config

ray.init()
tune.run(
    "PPO",
    config=dict(
        config,
        **{
            "env": ALPKukaEnv,
        }
    ),
    checkpoint_freq=10,
)
