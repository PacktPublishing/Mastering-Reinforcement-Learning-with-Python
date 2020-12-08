from tic_tac_toe import TicTacToe
import ray
from ray.rllib.agents.dqn.apex import ApexTrainer
import numpy as np
import random
import time
from ray.rllib.agents.ppo.ppo import PPOTrainer

if __name__ == "__main__":
    env = TicTacToe()
    ray.init()
    num_policies = 4
    policies = {
        "policy_{}".format(i): (None, env.observation_space, env.action_space, {})
        for i in range(num_policies)
    }
    policy_ids = list(policies.keys())
    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: random.choice(policy_ids)),
        },
        "framework": "tf",
    }
    #trainer = ApexTrainer(env=TicTacToe, config=config)
    trainer = PPOTrainer(env=TicTacToe, config=config)
    trainer.restore("ttt_model/checkpoint_119/checkpoint-119")
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        env.render()
        player = list(obs)[0]
        if player == "X":
            action = trainer.compute_action(np.array(obs["X"]), policy_id="policy_1")
        else:
            action = trainer.compute_action(np.array(obs["O"]), policy_id="policy_1")
        time.sleep(5)
        obs, rewards, dones, infos = env.step({player: action})
        done = dones["__all__"]
        print(obs, rewards, dones, infos)
    env.render()
