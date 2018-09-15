import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math
import threading
import random

from rollout_thread import RolloutThread
from env.terrain import Terrain
from random import randint
from env.sxsy import SXSY

class Rollout(object):
	
	def __init__(
		self,
		number_episode,
		num_task,
		map_index,
		sort_init):
		
		self.number_episode = number_episode
		self.map_index = map_index
		self.init_maps = SXSY[self.map_index]

		if sort_init == 'local':
			s = [l for ep in SXSY[self.map_index] for l in sorted(ep, key=lambda element: (element[1], element[0]))]
			s = np.array(s).astype(int)
			self.init_maps = s.reshape(1000, 20, 2)

		elif sort_init == 'global':
			s = [l for ep in SXSY[self.map_index] for l in ep]
			s = sorted(s, key=lambda element: (element[1], element[0]))
			s = np.array(s).astype(int)
			self.init_maps = s.reshape(1000, 20, 2)

		self.num_task = num_task
		self.states, self.tasks, self.actions, self.rewards = [self.holder_factory(self.num_task) for i in range(4)]

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

	def holder_factory(self, size):
		return [ [] for i in range(size) ]

	def rollout_batch(self, sess, network, policy, epoch):
		self.states, self.tasks, self.actions, self.rewards = [self.holder_factory(self.num_task) for i in range(4)]
		train_threads = []
		
		for i in range(self.number_episode):
			[sx, sy] = self.init_maps[epoch % 1000][i]
			for task in range(self.num_task):
				train_threads.append(threading.Thread(target=self._rollout_process, args=(sess, network, task, sx, sy, policy,)))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards	