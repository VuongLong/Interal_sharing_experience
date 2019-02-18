import numpy as np           # Handle matrices
import sys

from env.constants import TASK_LIST
from env.scene_loader import THORDiscreteEnvironment

class RolloutThread(object):
	
	def __init__(
		self,
		task,
		policy,
		scene_name):
	
		self.task = task
		self.policy = policy

		self.env = THORDiscreteEnvironment({"scene_name":scene_name, "terminal_state_id":[TASK_LIST[scene_name][task]]})

	def rollout(self):
		states, tasks, actions, rewards, next_states = [], [], [], [], []
		
		self.env.reset()
		state = self.env.current_state_id
		start = state
		step = 0	

		while True:
	
			action = np.random.choice(range(len(self.policy[state, self.task])), 
										  p=np.array(self.policy[state, self.task])/sum(self.policy[state, self.task]))  # select action w.r.t the actions prob

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

			if step > 50:
				break

		redundant_steps = step + self.env.shortest_path_distances[state, self.env.target[self.task]] - self.env.shortest_path_distances[start, self.env.target[self.task]]

		return states, tasks, actions, rewards, next_states, redundant_steps