# test_model.py

import os
from stable_baselines3 import PPO
from sc2bot import MODEL_DIR_PATH, LOG_DIR_PATH, get_model_files
from sc2bot.bot_env import BotEnv


# Set environment variables:
LATEST_MODEL_FILES = get_model_files()
LATEST_MODEL = LATEST_MODEL_FILES[-1]

# Init environment:
env = BotEnv()

# Load model:
model = PPO.load(LATEST_MODEL)

# Play the game:
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
