from env.terrain import Terrain
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math
from rollout_thread import RolloutThread
import threading
import random
from random import randint
from env.sxsy import SXSY

class Rollout(object):
	
	def __init__(
		self,
		number_episode,
		map_index):
		
		self.number_episode = number_episode
		self.map_index = map_index
		self.env = Terrain(map_index)

		self.states, self.tasks, self.actions, self.rewards, self.next_states = [],[],[],[],[]

		for task in range(self.env.num_task):
			self.states.append([])
			self.tasks.append([])
			self.actions.append([])
			self.rewards.append([])
			self.next_states.append([])
			for i in range(self.number_episode):
				self.states[task].append([])
				self.tasks[task].append([])
				self.actions[task].append([])
				self.rewards[task].append([])
				self.next_states[task].append([])


	def _rollout_process(self, sess, index, network, task, sx, sy, current_policy):
		thread_rollout = RolloutThread(
									sess = sess,
									network = network,
									task = task,
									start_x = sx,
									start_y = sy,
									policy = current_policy,
									map_index = self.map_index)

		ep_states, ep_tasks, ep_actions, ep_rewards, ep_next_states = thread_rollout.rollout()
		
		self.states[task][index]=ep_states
		self.tasks[task][index]=ep_tasks
		self.actions[task][index]=ep_actions
		self.rewards[task][index]=ep_rewards
		self.next_states[task][index]=ep_next_states


	def rollout_batch(self, sess, network, policy, epoch):
		self.states, self.tasks, self.actions, self.rewards, self.next_states = [],[],[],[],[]
		for task in range(self.env.num_task):
			self.states.append([])
			self.tasks.append([])
			self.actions.append([])
			self.rewards.append([])
			self.next_states.append([])
			for i in range(self.number_episode):
				self.states[task].append([])
				self.tasks[task].append([])
				self.actions[task].append([])
				self.rewards[task].append([])
				self.next_states[task].append([])

		train_threads = []
		for task in range(self.env.num_task):
			'''
			sx = 0
			sy = 0
			while self.env.MAP[sy][sx]==0:    
				sx = randint(1,self.env.bounds_x[1]) 
				sy = randint(1,self.env.bounds_y[1]) 
			'''
			index = 0
			for i in range(self.number_episode):
				[sx, sy] = SXSY[self.map_index][epoch-1][i]
				train_threads.append(threading.Thread(target=self._rollout_process, args=(sess, index, network, task, sx, sy, policy,)))
				index+=1	

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards, self.next_states	