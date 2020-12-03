import datetime
import numpy as np
import gym
import ray
from actor import Actor
from replay import ReplayBuffer
from learner import Learner
from parameter_server import ParameterServer
import tensorflow as tf
tf.get_logger().setLevel('WARNING')

def get_env_parameters(config):
    env = gym.make(config["env"])
    config['obs_shape'] = env.observation_space.shape
    config['n_actions'] = env.action_space.n


def main(config, max_samples):
    get_env_parameters(config)
    log_dir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    config['log_dir'] = log_dir
    ray.init()
    parameter_server = ParameterServer.remote(config)
    replay_buffer = ReplayBuffer.remote(config)
    learner = Learner.remote(config,
                             replay_buffer,
                             parameter_server)
    training_actor_ids = []
    eval_actor_ids = []

    learner.start_learning.remote()

    # Create training actors
    for i in range(config["num_workers"]):
        eps = config["max_eps"] * i / config["num_workers"]
        actor = Actor.remote("train-" + str(i),
                             replay_buffer,
                             parameter_server,
                             config,
                             eps)
        actor.sample.remote()
        training_actor_ids.append(actor)

    # Create eval actors
    for i in range(config["eval_num_workers"]):
        eps = 0
        actor = Actor.remote("eval-" + str(i),
                             replay_buffer,
                             parameter_server,
                             config,
                             eps,
                             True)
        eval_actor_ids.append(actor)

    total_samples = 0
    best_eval_mean_reward = np.NINF
    eval_mean_rewards = []
    while total_samples < max_samples:
        tsid = replay_buffer.get_total_env_samples.remote()
        new_total_samples = ray.get(tsid)
        if (new_total_samples - total_samples
                >= config["timesteps_per_iteration"]):
            total_samples = new_total_samples
            print("Total samples:", total_samples)
            parameter_server.set_eval_weights.remote()
            eval_sampling_ids = []
            for eval_actor in eval_actor_ids:
                sid = eval_actor.sample.remote()
                eval_sampling_ids.append(sid)
            eval_rewards = ray.get(eval_sampling_ids)
            print("Evaluation rewards: {}".format(eval_rewards))
            eval_mean_reward = np.mean(eval_rewards)
            eval_mean_rewards.append(eval_mean_reward)
            print("Mean evaluation reward: {}".format(eval_mean_reward))
            tf.summary.scalar('Mean evaluation reward', data=eval_mean_reward, step=total_samples)
            if eval_mean_reward > best_eval_mean_reward:
                print("Model has improved! Saving the model!")
                best_eval_mean_reward = eval_mean_reward
                parameter_server.save_eval_weights.remote()

    print("Finishing the training.")
    for actor in training_actor_ids:
        actor.stop.remote()
    learner.stop.remote()


if __name__ == '__main__':
    max_samples = 500000
    config = {"env": "CartPole-v0",
              "num_workers": 50,
              "eval_num_workers": 10,
              "n_step": 3,
              "max_eps": 0.5,
              "train_batch_size": 512,
              "gamma": 0.99,
              "fcnet_hiddens": [256, 256],
              "fcnet_activation": "tanh",
              "lr": 0.0001,
              "buffer_size": 1000000,
              "learning_starts": 5000,
              "timesteps_per_iteration": 10000,
              "grad_clip": 10}
    main(config, max_samples)
