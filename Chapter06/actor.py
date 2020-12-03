from collections import deque
import ray
import gym
import numpy as np
from models import get_Q_network


@ray.remote
class Actor:
    def __init__(self,
                 actor_id,
                 replay_buffer,
                 parameter_server,
                 config,
                 eps,
                 eval=False):
        self.actor_id = actor_id
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.config = config
        self.eps = eps
        self.eval = eval
        self.Q = get_Q_network(config)
        self.env = gym.make(config["env"])
        self.local_buffer = []
        self.obs_shape = config["obs_shape"]
        self.n_actions = config["n_actions"]
        self.multi_step_n = config.get("n_step", 1)
        self.q_update_freq = config.get("q_update_freq", 100)
        self.send_experience_freq = \
                    config.get("send_experience_freq", 100)
        self.continue_sampling = True
        self.cur_episodes = 0
        self.cur_steps = 0

    def update_q_network(self):
        if self.eval:
            pid = \
            self.parameter_server.get_eval_weights.remote()
        else:
            pid = \
            self.parameter_server.get_weights.remote()
        new_weights = ray.get(pid)
        if new_weights:
            self.Q.set_weights(new_weights)
        else:
            print("Weights are not available yet, skipping.")

    def get_action(self, observation):
        observation = observation.reshape((1, -1))
        q_estimates = self.Q.predict(observation)[0]
        if np.random.uniform() <= self.eps:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(q_estimates)
        return action

    def get_n_step_trans(self, n_step_buffer):
        gamma = self.config['gamma']
        discounted_return = 0
        cum_gamma = 1
        for trans in list(n_step_buffer)[:-1]:
            _, _, reward, _ = trans
            discounted_return += cum_gamma * reward
            cum_gamma *= gamma
        observation, action, _, _ = n_step_buffer[0]
        last_observation, _, _, done = n_step_buffer[-1]
        experience = (observation, action, discounted_return,
                      last_observation, done, cum_gamma)
        return experience

    def stop(self):
        self.continue_sampling = False

    def sample(self):
        print("Starting sampling in actor {}".format(self.actor_id))
        self.update_q_network()
        observation = self.env.reset()
        episode_reward = 0
        episode_length = 0
        n_step_buffer = deque(maxlen=self.multi_step_n + 1)
        while self.continue_sampling:
            action = self.get_action(observation)
            next_observation, reward, \
            done, info = self.env.step(action)
            n_step_buffer.append((observation, action,
                                  reward, done))
            if len(n_step_buffer) == self.multi_step_n + 1:
                self.local_buffer.append(
                    self.get_n_step_trans(n_step_buffer))
            self.cur_steps += 1
            episode_reward += reward
            episode_length += 1
            if done:
                if self.eval:
                    break
                next_observation = self.env.reset()
                if len(n_step_buffer) > 1:
                    self.local_buffer.append(
                        self.get_n_step_trans(n_step_buffer))
                self.cur_episodes += 1
                episode_reward = 0
                episode_length = 0
            observation = next_observation
            if self.cur_steps % \
                    self.send_experience_freq == 0 and not self.eval:
                self.send_experience_to_replay()
            if self.cur_steps % \
                    self.q_update_freq == 0 and not self.eval:
                self.update_q_network()
        return episode_reward

    def send_experience_to_replay(self):
        rf = self.replay_buffer.add.remote(self.local_buffer)
        ray.wait([rf])
        self.local_buffer = []
