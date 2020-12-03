from collections import deque
import ray
import numpy as np


@ray.remote
class ReplayBuffer:
    def __init__(self, config):
        self.replay_buffer_size = config["buffer_size"]
        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.total_env_samples = 0

    def add(self, experience_list):
        experience_list = experience_list
        for e in experience_list:
            self.buffer.append(e)
            self.total_env_samples += 1
        return True

    def sample(self, n):
        if len(self.buffer) > n:
            sample_ix = np.random.randint(
                len(self.buffer), size=n)
            return [self.buffer[ix] for ix in sample_ix]

    def get_total_env_samples(self):
        return self.total_env_samples