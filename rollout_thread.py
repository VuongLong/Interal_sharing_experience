import sys
import numpy as np           		# Handle matrices
import random                		# Handling random number generation
import time                  		# Handling time calculation
import math

from env.terrain import Terrain
from collections import deque		# Ordered collection with ends

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
		states, tasks, actions, rewards_of_episode = [], [], [], []
		
		self.env.resetgame(self.task, self.start_x, self.start_y)
		state = self.env.player.getposition()

		step = 1	

		while True:
			step+=1
			if step > 20000:
				print('re-rollout')
				sys.stdout.flush()
				break

			pi = self.policy[state[0], state[1], self.task]
			action = np.random.choice(range(len(pi)), p =np.array(pi)/sum(pi))  # select action w.r.t the actions prob

			reward, done = self.env.player.action(action)
			
			next_state = self.env.player.getposition()
			
			# Store results
			states.append(state)
			tasks.append(self.task)

			actions.append(action)
			rewards_of_episode.append(reward)
			state = next_state
			
			if done:     
				break

		return states, tasks, actions, rewards_of_episode	