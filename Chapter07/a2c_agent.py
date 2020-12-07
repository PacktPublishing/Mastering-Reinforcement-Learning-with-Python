import argparse
import pprint
from ray import tune
import ray
from ray.rllib.agents.a3c.a2c import (
    A2C_DEFAULT_CONFIG as DEFAULT_CONFIG,
    A2CTrainer)


if __name__ == "__main__":
    trainer = A2CTrainer
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        help='Gym env name.')
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    config_update = {
                 "env": args.env,
                 "num_gpus": 1,
                 "num_workers": 50,
                 "evaluation_num_workers": 10,
                 "evaluation_interval": 1,
                 "use_gae": False
            }
    config.update(config_update)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    ray.init()
    tune.run(trainer,
             stop={"timesteps_total": 2000000},
             config=config
             )
