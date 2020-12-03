import pprint
from ray import tune
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.dqn.apex import ApexTrainer

if __name__ == '__main__':
    config = APEX_DEFAULT_CONFIG.copy()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    config['env'] = "CartPole-v0"
    config['num_workers'] = 50
    config['evaluation_num_workers'] = 10
    config['evaluation_interval'] = 1
    config['learning_starts'] = 5000
    tune.run(ApexTrainer, config=config)
