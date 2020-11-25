import csv
from datetime import datetime
import numpy as np
import ray
from ray.tune.logger import pretty_print
from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog

from custom_mcar import MountainCar
from masking_model import ParametricActionsModel
from mcar_demo import DEMO_DATA_DIR

ALL_STRATEGIES = [
    "default",
    "with_dueling",
    "custom_reward",
    "custom_reward_n_dueling",
    "demonstration",
    "curriculum",
    "curriculum_n_dueling",
    "action_masking",
]
STRATEGY = "demonstration"
CURRICULUM_MAX_LESSON = 4
CURRICULUM_TRANS = 150
MAX_STEPS = 2e6
MAX_STEPS_OFFLINE = 4e5
NUM_TRIALS = 5
NUM_FINAL_EVAL_EPS = 20


def get_apex_trainer(strategy):
    config = APEX_DEFAULT_CONFIG.copy()
    config["env"] = MountainCar
    config["buffer_size"] = 1000000
    config["learning_starts"] = 10000
    config["target_network_update_freq"] = 50000
    config["rollout_fragment_length"] = 200
    config["timesteps_per_iteration"] = 10000
    config["num_gpus"] = 1
    config["num_workers"] = 20
    config["evaluation_num_workers"] = 10
    config["evaluation_interval"] = 1
    if strategy not in [
        "with_dueling",
        "custom_reward_n_dueling",
        "curriculum_n_dueling",
    ]:
        config["hiddens"] = []
        config["dueling"] = False

    if strategy == "action_masking":
        ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
        config["env_config"] = {"use_action_masking": True}
        config["model"] = {
            "custom_model": "pa_model",
        }
    elif strategy == "custom_reward" or strategy == "custom_reward_n_dueling":
        config["env_config"] = {"reward_fun": "custom_reward"}
    elif strategy in ["curriculum", "curriculum_n_dueling"]:
        config["env_config"] = {"lesson": 0}
    elif strategy == "demonstration":
        config["input"] = DEMO_DATA_DIR
        #config["input"] = {"sampler": 0.7, DEMO_DATA_DIR: 0.3}
        config["explore"] = False
        config["input_evaluation"] = []
        config["n_step"] = 1

    trainer = ApexTrainer(config=config)
    return trainer, config["env_config"]


def set_trainer_lesson(trainer, lesson):
    trainer.evaluation_workers.foreach_worker(
        lambda ev: ev.foreach_env(lambda env: env.set_lesson(lesson))
    )
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(lambda env: env.set_lesson(lesson))
    )


def increase_lesson(lesson):
    if lesson < CURRICULUM_MAX_LESSON:
        lesson += 1
    return lesson


def final_evaluation(trainer, n_final_eval, env_config={}):
    env = MountainCar(env_config)
    eps_lengths = []
    for i_episode in range(n_final_eval):
        observation = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            action = trainer.compute_action(observation)
            observation, reward, done, info = env.step(action)
            if done:
                eps_lengths.append(t)
                print(f"Episode finished after {t} time steps")
    print(
        f"Avg. episode length {np.mean(eps_lengths)} out of {len(eps_lengths)} episodes."
    )
    return np.mean(eps_lengths)


### START TRAINING ###
ray.init()
avg_eps_lens = []
for i in range(NUM_TRIALS):
    trainer, env_config = get_apex_trainer(STRATEGY)
    if STRATEGY in ["curriculum", "curriculum_n_dueling"]:
        lesson = 0
        set_trainer_lesson(trainer, lesson)
    # Training
    while True:
        results = trainer.train()
        print(pretty_print(results))
        if STRATEGY == "demonstration":
            demo_training_steps = results["timesteps_total"]
            if results["timesteps_total"] >= MAX_STEPS_OFFLINE:
                trainer, _ = get_apex_trainer("with_dueling")
        if results["timesteps_total"] >= MAX_STEPS:
            if STRATEGY == "demonstration":
                if results["timesteps_total"] >= MAX_STEPS + demo_training_steps:
                    break
            else:
                break
        if "evaluation" in results and STRATEGY in ["curriculum", "curriculum_n_dueling"]:
            if results["evaluation"]["episode_len_mean"] < CURRICULUM_TRANS:
                lesson = increase_lesson(lesson)
                set_trainer_lesson(trainer, lesson)
                print(f"Lesson: {lesson}")

    # Final evaluation
    checkpoint = trainer.save()
    if STRATEGY in ["curriculum", "curriculum_n_dueling"]:
        env_config["lesson"] = CURRICULUM_MAX_LESSON
    if STRATEGY == "action_masking":
        # Action masking is running into errors in Ray 1.0.1 during compute action
        # So, we use evaluation episode lengths.
        avg_eps_len = results["evaluation"]["episode_len_mean"]
    else:
        avg_eps_len = final_evaluation(trainer, NUM_FINAL_EVAL_EPS, env_config)
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    result = [date_time, STRATEGY, str(i), avg_eps_len, checkpoint]
    avg_eps_lens.append(avg_eps_len)
    with open(r"results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(result)
print(f"Average episode length: {np.mean(avg_eps_lens)}")

