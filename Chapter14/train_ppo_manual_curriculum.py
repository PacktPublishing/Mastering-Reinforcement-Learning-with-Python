import ray
from ray import tune

from custom_kuka import CustomKukaEnv
from configs import config

config["env_config"] = {
    "isDiscrete": False,
    "renders": False,
    "jp_override": {"2": 1.3, "4": -1},
    "rnd_obj_x": 0,
    "rnd_obj_y": 0,
    "rnd_obj_ang": 0,
    "bias_obj_x": 0,
    "bias_obj_y": 0.04,
    "bias_obj_ang": 0,
}


def on_train_result(info):
    result = info["result"]
    if result["episode_reward_mean"] > 5.5:
        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(lambda env: env.increase_difficulty())
        )


ray.init()
tune.run(
    "PPO",
    config=dict(
        config,
        **{
            "env": CustomKukaEnv,
            "callbacks": {
                "on_train_result": on_train_result,
            },
        }
    ),
    checkpoint_freq=10,
)
