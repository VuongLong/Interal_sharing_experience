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

		self.states, self.tasks, self.actions, self.rewards = [[],[]], [[],[]], [[],[]], [[],[]]


	def _rollout_process(self, sess, network, task, sx, sy, current_policy):
		thread_rollout = RolloutThread(
									sess = sess,
									network = network,
									task = task,
									start_x = sx,
									start_y = sy,
									policy = current_policy,
									map_index = self.map_index)

		ep_states, ep_tasks, ep_actions, ep_rewards = thread_rollout.rollout()
		
		self.states[task].append(ep_states)
		self.tasks[task].append(ep_tasks)
		self.actions[task].append(ep_actions)
		self.rewards[task].append(ep_rewards)


	def rollout_batch(self, sess, network, policy, epoch):
		self.states, self.tasks, self.actions, self.rewards = [[],[]], [[],[]], [[],[]], [[],[]]
		train_threads = []
		for i in range(self.number_episode):
			[sx, sy] = SXSY[self.map_index][epoch-1][i]
			for task in range(self.env.num_task):
				train_threads.append(threading.Thread(target=self._rollout_process, args=(sess, network, task, sx, sy, policy,)))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards	