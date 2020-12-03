import time
import ray
import numpy as np
from models import get_trainable_model
from tensorflow.keras.models import clone_model


@ray.remote
class Learner:
    def __init__(self, config, replay_buffer, parameter_server):
        self.config = config
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.Q, self.trainable = get_trainable_model(config)
        self.target_network = clone_model(self.Q)
        self.train_batch_size = config["train_batch_size"]
        self.total_collected_samples = 0
        self.samples_since_last_update = 0
        self.send_weights_to_parameter_server()
        self.stopped = False

    def send_weights_to_parameter_server(self):
        self.parameter_server.update_weights.remote(self.Q.get_weights())

    def start_learning(self):
        print("Learning starting...")
        self.send_weights()
        while not self.stopped:
            sid = self.replay_buffer.get_total_env_samples.remote()
            total_samples = ray.get(sid)
            if total_samples >= self.config["learning_starts"]:
                self.optimize()

    def optimize(self):
        samples = ray.get(self.replay_buffer
                          .sample.remote(self.train_batch_size))
        if samples:
            N = len(samples)
            self.total_collected_samples += N
            self.samples_since_last_update += N
            ndim_obs = 1
            for s in self.config["obs_shape"]:
                if s:
                    ndim_obs *= s
            n_actions = self.config["n_actions"]
            obs = np.array([sample[0] for sample \
                        in samples]).reshape((N, ndim_obs))
            actions = np.array([sample[1] for sample \
                        in samples]).reshape((N,))
            rewards = np.array([sample[2] for sample \
                        in samples]).reshape((N,))
            last_obs = np.array([sample[3] for sample \
                        in samples]).reshape((N, ndim_obs))
            done_flags = np.array([sample[4] for sample \
                        in samples]).reshape((N,))
            gammas = np.array([sample[5] for sample \
                        in samples]).reshape((N,))
            masks = np.zeros((N, n_actions))
            masks[np.arange(N), actions] = 1
            dummy_labels = np.zeros((N,))
            # double DQN
            maximizer_a = np.argmax(self.Q.predict(last_obs),
                                    axis=1)
            target_network_estimates = \
                self.target_network.predict(last_obs)
            q_value_estimates = \
                np.array([target_network_estimates[i,
                                      maximizer_a[i]]
                        for i in range(N)]).reshape((N,))
            sampled_bellman = rewards + gammas * \
                              q_value_estimates * \
                              (1 - done_flags)
            trainable_inputs = [obs, masks,
                                sampled_bellman]
            self.trainable.fit(trainable_inputs,
                               dummy_labels, verbose=0)
            self.send_weights()

            if self.samples_since_last_update > 500:
                self.target_network.set_weights(self.Q.get_weights())
                self.samples_since_last_update = 0
            return True
        else:
            print("No samples received from the buffer.")
            time.sleep(5)
            return False

    def send_weights(self):
        id = self.parameter_server.update_weights.remote(self.Q.get_weights())
        ray.get(id)

    def stop(self):
        self.stopped = True
