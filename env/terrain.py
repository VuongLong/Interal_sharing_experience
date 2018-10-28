import matplotlib.pyplot as plt
import random
from random import randint
import numpy as np
from env.map import ENV_MAP
from constant import ACTION_SIZE
plt.ion()

from env.controller import Player

class Terrain:
	def __init__(self, map_index):
		self.reward_locs = ENV_MAP[map_index]['goal']
		self.MAP = ENV_MAP[map_index]['map']
		self.bounds_x = ENV_MAP[map_index]['size_x']
		self.bounds_y = ENV_MAP[map_index]['size_y']
		self.size_m = ENV_MAP[map_index]['size_m']

		self.action_size = ACTION_SIZE
		self.reward_range = 1.0
		self.reward_goal = 1.0
		
		self.num_task = len(self.reward_locs)

		self.cv_state_onehot = np.identity(self.bounds_x[1]*self.bounds_y[1],dtype=int)
		self.cv_action_onehot = np.identity(self.action_size,dtype=int)
		self.cv_task_onehot = np.identity(len(self.reward_locs),dtype=int)
		


	def getreward(self):
		done = False
		reward = -0.01

		x_pos, y_pos = self.reward_locs[self.task]
		#reward -= 0.15
		if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
			reward = self.reward_goal
			done = True

		return reward, done

	def checkepisodeend(self):
		for x_pos, y_pos in self.reward_locs:
			if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
				return 1
		return 0

	def plotgame(self):
		plt.clf()
		for x_pos, y_pos in self.reward_locs:
			plt.plot([x_pos,], [y_pos,], marker='o', markersize=10, color="green")
		plt.xlim([self.bounds_x[0]-1,self.bounds_x[1]+1])
		plt.ylim([self.bounds_y[0]-1,self.bounds_y[1]+1])

		for y in range(self.bounds_y[0]-1,self.bounds_y[1]+2):
			for x in range(self.bounds_x[0]-1,self.bounds_x[1]+2):
				if MAP[y][x]==0:
					plt.plot([x,], [y,], marker='o', markersize=2, color="green")

		plt.plot([self.player.x,], [self.player.y,], marker='x', markersize=10, color="red")
		plt.pause(0.001)

	def resetgame(self, task, sx, sy):
		self.player = Player(sx, sy, self)

		self.task = task
			
	def caculate_minimum_steps(self):
		def step(action_size, action, x, y):
			if action_size == 8:
				cv_action = [[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]]
			else:
				cv_action = [[0,-1],[-1,0],[0,1],[1,0]]
			
			new_x = x + cv_action[action][0]
			new_y = y + cv_action[action][1]

			if(self.MAP[new_y][new_x] == 0):
				return -1
			else:
				return new_x, new_y
		
		def to_np(min_step_dict, bounds_x, bounds_y):
			result = np.full((bounds_y[1] + 2, bounds_x[1] + 2), -1)
			for x in range(bounds_x[1] + 2):
				for y in range(bounds_y[1] + 2):
					result[y][x] = min_step_dict.get((x, y), -1)
			return result


		self.min_step = {}
		visit_queue = {}
		for task_idx in range(self.num_task):
			self.min_step[task_idx] = {}
			self.min_step[task_idx][self.reward_locs[task_idx][0], self.reward_locs[task_idx][1]] = 0
			visit_queue = []
			visit_queue.append((self.reward_locs[task_idx][0], self.reward_locs[task_idx][1], 0))
			while(len(visit_queue) != 0):
				x, y, dist = visit_queue[0]
				# print(x, y)
				visit_queue = visit_queue[1:]
				for action in range(self.action_size):
					step_result = step(self.action_size, action, x, y)
					if step_result == -1:
						continue
					else:
						new_x, new_y = step_result
						if (new_x, new_y) not in self.min_step[task_idx]:
							self.min_step[task_idx][new_x, new_y] = dist + 1
							visit_queue.append((new_x, new_y, dist + 1))
		
		# for task_idx in range(self.num_task):
		# 	print("Step map task {}:".format(task_idx))
		# 	print(to_np(self.min_step[task_idx], self.bounds_x, self.bounds_y))