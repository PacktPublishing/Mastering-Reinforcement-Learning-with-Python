#!/usr/bin/env python
import os

import numpy as np
import sys, gym, time

import ray.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from custom_mcar import MountainCar

DEMO_DATA_DIR = "mcar-out"


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xFF0D:
        human_wants_restart = True
    if key == 32:
        human_sets_pause = not human_sets_pause
    a = int(key - ord("0"))
    if a <= 0 or a >= ACTIONS:
        return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord("0"))
    if a <= 0 or a >= ACTIONS:
        return
    if human_agent_action == a:
        human_agent_action = 0


def rollout(env, eps_id):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obs = env.reset()
    prev_action = np.zeros_like(env.action_space.sample())
    prev_reward = 0
    t = 0
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        new_obs, r, done, info = env.step(a)
        # Build the batch
        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            obs=prep.transform(obs),
            actions=a,
            action_prob=1.0,  # put the true action probability here
            action_logp=0,
            action_dist_inputs=None,
            rewards=r,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos=info,
            new_obs=prep.transform(new_obs),
        )
        obs = new_obs
        prev_action = a
        prev_reward = r

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.wrapped.render()
        if window_still_open == False:
            return False
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.wrapped.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    writer.write(batch_builder.build_and_reset())


if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(DEMO_DATA_DIR)

    env = MountainCar()

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    if not hasattr(env.action_space, "n"):
        raise Exception("Keyboard agent only supports discrete action spaces")
    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
    # can test what skip is still usable.

    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False

    env.reset()
    env.wrapped.render()
    env.wrapped.unwrapped.viewer.window.on_key_press = key_press
    env.wrapped.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    for i in range(20):
        window_still_open = rollout(env, i)
        if window_still_open == False:
            break

