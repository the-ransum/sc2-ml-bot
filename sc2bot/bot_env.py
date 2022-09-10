# bot_env.py

import os
import subprocess
import pickle
import gym
import numpy as np
from gym import spaces
from sc2bot import PICKLE_FILE_PATH


class BotEnv(gym.Env):
	""" Customized environment for following the gym interface """
    
	def __init__(self):
		super(BotEnv, self).__init__()
        
		self.PICKLE_FILE_PATH = PICKLE_FILE_PATH
        
		# Define action and observation space they must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(224, 224, 3),
			dtype=np.uint8
		)
	
	def step(self, action):
		wait_for_action = True
		
		# waits for action.
		while wait_for_action:
			# print("[!] Awaiting action...")
			try:
				with open(self.PICKLE_FILE_PATH, 'rb') as f:
					state_rwd_action = pickle.load(f)
					
					if state_rwd_action['action'] is not None:
						# print("[!] No action found.")
						wait_for_action = True
					else:
						# print("[!] Action found.")
						wait_for_action = False
						state_rwd_action['action'] = action
						with open(self.PICKLE_FILE_PATH, 'wb') as f:
							# print("[!] Writing action.")
							pickle.dump(state_rwd_action, f)
			except Exception as e:
				# print("[X] Step Action - Exception: ", str(e))
				pass
		
		# await for new state to return for map and reward, no new action yet.
		wait_for_state = True
		while wait_for_state:
			try:
				if os.path.getsize(self.PICKLE_FILE_PATH) > 0:
					with open(self.PICKLE_FILE_PATH, 'rb') as f:
						state_rwd_action = pickle.load(f)
						if state_rwd_action['action'] is None:
							# print("[!] No state yet")
							wait_for_state = True
						else:
							# print("[!] State: ", state_rwd_action['state'])
							state = state_rwd_action['state']
							reward = state_rwd_action['reward']
							done = state_rwd_action['done']
							wait_for_state = False
							
			except Exception as e:
				wait_for_state = True   
				map = np.zeros((224, 224, 3), dtype=np.uint8)
				observation = map
				
				# empty state waiting for the next one, input an action "3 - scout".
				data = {"state": map, 
						"reward": 0, 
						"action": 3, 
						"done": False}
					
				with open(self.PICKLE_FILE_PATH, 'wb') as f:
					pickle.dump(data, f)
					
				state, reward, done, action = map, 0, False, 3
				
		info = {}
		observation = state
		return observation, reward, done, info
	
	def reset(self):
		print("[!] Environment reset.")
		
		map = np.zeros((224, 224, 3), dtype=np.uint8)
		observation = map
		
		# empty state waiting for the next one.
		data = {"state": map, 
				"reward": 0, 
				"action": None, 
				"done": False}
			
		with open(self.PICKLE_FILE_PATH, 'wb') as f:
			pickle.dump(data, f)
			
		# start "ai-trainer.py" in a new process, non-blocking.
		subprocess.Popen(['python3', 'ai-trainer.py'])
		
		# reward and done, info can't be included
		return observation