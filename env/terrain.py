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
			
