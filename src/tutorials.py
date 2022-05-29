"""KNN Project 2022 - tutorials.py

This module implements tutorials from stable_baselines3 and gym documentations.
"""

__author__ = "Martin Kostelník (xkoste12), Michal Glos (xglosm01), Michal Szymik (xszymi00)"

import gym
from gym.wrappers import Monitor

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_FOLDER = "models"
VIDEO_FOLDER = "videos"

def baselines_a2c_pendulum():
    env = gym.make("InvertedPendulum-v2")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    obs = env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            obs = env.reset()

    env.close()


def baselines_a2c_cheetah():
    env = gym.make("HalfCheetah-v2")

    model = A2C("MlpPolicy", env, verbose=1, use_rms_prop=False)
    model.learn(total_timesteps=100000)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    obs = env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            obs = env.reset()

    env.close()

# https://github.com/zuoxingdong/dm2gym
def baselines_a2c_hopper():
    env = gym.make('dm2gym:HopperStand-v0', visualize_reward=True)
    env = Monitor(env, f"./{VIDEO_FOLDER}", video_callable=lambda episode_id: episode_id % 10 == 0, force=True)

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=3000)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    obs = env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            obs = env.reset()

    env.close()
