# train_model_mlpp.py

import os
import time
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO

from sc2bot import MODEL_DIR_PATH, LOGS_DIR_PATH, get_model_files
from sc2bot.bot_env import BotEnv


# Build environment:
model_name = str(int(time()))
models_dir = os.path.join(MODEL_DIR_PATH, model_name)
logs_dir = os.path.join(LOGS_DIR_PATH, model_name)

# Initialize wandb:
conf_dict = {"Model": "v19",
            "Machine": "Main",
            "policy": "MlpPolicy",
            "model_save_name": model_name}

run = wandb.init(
    project="sc2botv1",
    entity="ransum",
    config=conf_dict,
    # auto-upload sb3 tensorboard metrics
    sync_tensorboard=True,
    # save source code
    save_code=True,
)

# Init environment:
env = BotEnv()
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)

# Train model:
TIMESTEPS = 10000
iters = 0
while True:
	print("Frame: %i" % iters)
	iters += 1
	model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        tb_log_name="PPO"
    )
	model.save(os.path.join(models_dir, str(TIMESTEPS*iters)))
