from env.terrain import Terrain
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
import sys

class RolloutThread(object):
	
	def __init__(
		self,
		sess,
		network,
		task,
		start_x,
		start_y,
		policy,
		map_index):
	
		self.sess = sess
		self.network = network
		self.task = task
		self.start_x = start_x
		self.start_y = start_y
		self.policy = policy
		self.env = Terrain(map_index)
		self.onehot_actions = np.identity(self.env.action_size, dtype=int)

	def rollout(self):
		states, tasks, actions, rewards, next_states = [], [], [], [], []
		
		self.env.resetgame(self.task, self.start_x, self.start_y)
		state = self.env.player.getposition()

		step = 1	

		while True:
			if step > 50:
				#print('re-rollout')
				#sys.stdout.flush()
				break
				#states, tasks, actions, rewards = [], [], [], []
				#step = 1
				#self.env.resetgame(self.task, self.start_x, self.start_y)
				#state = self.env.player.getposition()


			action = np.random.choice(range(len(self.policy[state[0],state[1],self.task])), 
										  p=np.array(self.policy[state[0],state[1],self.task])/sum(self.policy[state[0],state[1],self.task]))  # select action w.r.t the actions prob

			reward, done = self.env.player.action(action)
			
			next_state = self.env.player.getposition()
			
			# Store results
			states.append(state)
			tasks.append(self.task)

			actions.append(action)
			rewards.append(reward)
			next_states.append(next_state)
			state = next_state
			
			if done:     
				break

			step+=1

		redundant_steps = step + self.env.min_step[self.task][states[-1][0], states[-1][1]] - self.env.min_step[self.task][self.start_x, self.start_y]

		return states, tasks, actions, rewards, next_states, redundant_steps