import numpy as np
import random
import sys

from utils import noise_and_argmax
from env.scene_loader import PyGameDumpEnv

class RolloutThread(object):
	
	def __init__(
		self,
		task,
		num_step,
		policy,
		scene_name):
		
		self.num_step = num_step
		self.task = task
		self.policy = policy

		self.env = PyGameDumpEnv({"scene_name":scene_name,
								"task": task,
								"anti_collision": 1,
								"success_reward": 1.0})

	def rollout(self, epsilon):
		states, tasks, actions, rewards, next_states = [], [], [], [], []
		
		self.env.reset()
		state = self.env.current_state_id
		start = state
		step = 0	

		while True:
	
			rand = random.random()
			if rand < epsilon:
				action = np.random.choice(range(self.env.action_size))
			else:
				try:
					action = np.random.choice(range(len(self.policy[state, self.task])), 
												  p=np.array(self.policy[state, self.task])/sum(self.policy[state, self.task]))  # select action w.r.t the actions prob
				except ValueError:
					print(self.policy[state, self.task])
					sys.exit()
				
			self.env.step(action)
			
			next_state = self.env.current_state_id
			
			# Store results
			states.append(state)
			tasks.append(self.task)

			actions.append(action)
			rewards.append(self.env.reward)
			next_states.append(next_state)
			state = next_state
			
			if self.env.terminal:     
				break

			step+=1

			if step > self.num_step:
				break

		# redundant_steps = step + self.env.shortest_path_distances[state, self.env.target[self.task]] - self.env.shortest_path_distances[start, self.env.target[self.task]]
		redundant_steps = 0

		return states, tasks, actions, rewards, next_states, redundant_steps