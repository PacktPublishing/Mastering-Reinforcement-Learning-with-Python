import argparse
import pprint
from ray import tune
import ray
from ray.rllib.agents.sac.sac import (
    DEFAULT_CONFIG,
    SACTrainer as trainer)


if __name__ == "__main__":
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
                 "evaluation_interval": 1
            }
    config.update(config_update)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    ray.init()
    tune.run(trainer,
             stop={"timesteps_total": 2000000},
             config=config
             )
