# -*- coding: utf-8 -*-
"""
Local Version of Reinforcement Learning - Breakout

This version allows you to train and save the model directly on your local computer.
Simply run the program and let it reach the timestep below

*NOTE - You might have to train the model in Python 3.11, I'm not sure if this works on
later versions due to some packages not updated to later versions.
"""

training_timesteps = 2_000_000

# Project Dependencies
import warnings
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import matplotlib.pyplot as plt
    import imageio
    import gradio as gr
    import ale_py

    from PIL import Image
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecVideoRecorder
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    gym.register_envs(ale_py)

print("All dependencies loaded successfully!")

# Save directory (change this to wherever you want)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SAVE_DIR, "breakout_ppo")

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
obs, info = env.reset()

frames = []

for _ in range(200):
    action = env.action_space.sample()
    obs, _, done, truncated, _ = env.step(action)
    frames.append(obs)
    if done or truncated:
        obs, info = env.reset()

env.close()

gif_path = os.path.join(SAVE_DIR, "test_run.gif")
imageio.mimsave(gif_path, frames, duration=30)
Image.open(gif_path).show()

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env("ALE/Breakout-v5", n_envs=8, seed=0)
env = VecFrameStack(env, n_stack=4)

model = PPO(
    "CnnPolicy",    
    env,
    learning_rate=2.5e-4,
    n_steps=256,
    batch_size=512,
    n_epochs=4,
    gamma=0.99,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

model.learn(total_timesteps=training_timesteps)
model.save(MODEL_PATH)
print("Training complete!")

model = PPO.load(MODEL_PATH)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

obs = eval_env.reset()
frames = []
total_reward = 0

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    frame = eval_env.render()
    frames.append(frame)
    total_reward += reward[0]
    if done[0]:
        break

eval_env.close()

demo_path = os.path.join(SAVE_DIR, "breakout_demo.gif")
imageio.mimsave(demo_path, frames, duration=10)
print(f"Total reward: {total_reward}")
Image.open(demo_path).show()

