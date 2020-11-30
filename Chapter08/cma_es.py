import copy
import gym
import cma
import numpy as np
import ray


@ray.remote
def rollout(env, actions, n_actions, look_ahead):
    env = copy.deepcopy(env)
    sampled_reward = 0
    for i in range(look_ahead):
        s_ix = i * n_actions
        e_ix = (i + 1) * n_actions
        action = actions[s_ix:e_ix]
        obs, reward, done, info = env.step(action)
        sampled_reward += reward
        if done:
            break
    return sampled_reward


class CMAESOptimizer:
    def __init__(self, env_name, look_ahead, num_ep, control, opt_iter=100):
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.shape[0]
        self.look_ahead = look_ahead
        self.n_tot_actions = self.look_ahead * self.n_actions
        self.num_ep = num_ep
        self.control = control
        self.episode_rewards = []
        self.num_episodes = 0
        self.opt_iter = opt_iter

    def optimize(self):
        # env.render()
        print("here 1")
        for i in range(self.num_ep):
            self.env.reset()
            ep_reward = 0
            done = False
            j = 0
            while not done:
                print(j)
                j += 1
                actions = self.cma_es_optimizer()
                if self.control == "open-loop":
                    for a in actions:
                        obs, reward, done, info = self.env.step(a)
                        ep_reward += reward
                        if done:
                            break
                elif self.control == "closed-loop":
                    obs, reward, done, info = self.env.step(actions[0])
                    ep_reward += reward
                else:
                    raise ValueError("Unknown control type.")
                if done:
                    self.episode_rewards.append(ep_reward)
                    self.num_episodes += 1
                    print(f"Episode {i}, reward: {ep_reward}")

    def cma_es_optimizer(self):
        es = cma.CMAEvolutionStrategy([0] \
                                      * self.n_tot_actions, 1)
        while (not es.stop()) and \
                es.result.iterations <= self.opt_iter:
            X = es.ask()  # get list of new solutions
            futures = [
                rollout.remote(self.env, x,
                               self.n_actions, self.look_ahead)
                for x in X
            ]
            costs = [-ray.get(id) for id in futures]
            es.tell(X, costs)  # feed values
            es.disp()
        actions = [
            es.result.xbest[i * self.n_actions : \
                            (i + 1) * self.n_actions]
            for i in range(self.look_ahead)
        ]
        return actions


if __name__ == "__main__":
    np.random.seed(0)
    ray.init()
    cma_opt = CMAESOptimizer(
        env_name="BipedalWalker-v3", look_ahead=10, num_ep=1, control="closed-loop"
    )
    cma_opt.optimize()
