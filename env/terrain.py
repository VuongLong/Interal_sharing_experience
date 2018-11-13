import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from random import randint
import numpy as np
import os
import sys
import tensorflow as tf
sys.path.insert(0, '../')
from network import ZNetwork
from map import ENV_MAP
from constant import ACTION_SIZE
plt.ion()

from controller import Player

class Terrain:
	def __init__(self, map_index):
		self.reward_locs = ENV_MAP[map_index]['goal']
		self.pretrain = ENV_MAP[map_index]['pretrain']
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
		
		self.zmap = {}
		self.visit = {}

		for i in range(self.num_task-1):
			for j in range(i+1,self.num_task):
				self.zmap[i, j] = np.zeros((len(self.MAP), len(self.MAP[0]))).astype(int)

		for i in range(self.num_task):
			self.visit[i] = np.zeros((len(self.MAP), len(self.MAP[0]))).astype(int)

		self.caculate_minimum_steps()

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
	
	def save_zmap(self, save_dir):
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)

		for i in range(self.num_task-1):
			for j in range(i+1,self.num_task):
				np.save(os.path.join(save_dir, "zmap_{}_{}".format(i, j)), self.zmap[i, j])

		for i in range(self.num_task):
			np.save(os.path.join(save_dir, "visit_{}".format(i)), self.visit[i])


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

	def plot_z(self, weights_path):
		
		plt.clf()
		num_plots = self.num_task * (self.num_task - 1) / 2
		num_rows = num_plots // 2 + num_plots % 2 
		num_cols = 2 if num_plots > 1 else 1
		index = 0
		scatter_size = 500
		for i in range(self.num_task - 1, 0, -1):
			for j in range(i-1, -1, -1):
				index += 1
				plt.subplot(num_rows, num_cols, index)
				plt.title('{}_{}'.format(j+1, i+1))
				
				plt.xlim([-1, len(self.MAP[0])])
				plt.ylim([-1, len(self.MAP)])
				plt.tick_params(
						axis='x',          # changes apply to the x-axis
						which='both',      # both major and minor ticks are affected
						bottom=False,      # ticks along the bottom edge are off
						top=False,
						left = False,         # ticks along the top edge are off
						labelbottom=False,
						)

				tf.reset_default_graph()
				sess = tf.Session()
				oracle = ZNetwork(
							state_size = self.size_m,
							action_size = 2,
							learning_rate = 0.005,
							name = 'oracle{}_{}'.format(j, i)
							)

				oracle.restore_model(sess, weights_path)

				zmap = np.zeros((len(self.MAP), len(self.MAP[0])))
				cm = plt.cm.get_cmap('jet')

				wx_s = []
				wy_s = []
				range_x = [0, self.bounds_x[1]+1]
				range_y = [0, self.bounds_y[1]+1]

		
				for x in range(range_x[0]+1,range_x[1]):
					for y in range(range_y[0]+1,range_y[1]):
						if self.MAP[y][x] == 0:
							wx_s.append(x)
							wy_s.append(y)
						# elif :
						#     plt.scatter(x, y, c = 'white', s = scatter_size)
						#     plt.text(x - 0.3, y - 0.3 , s = self.MAP[y][x], fontsize = 20)            
						else:
							state_index = x+(y-1)*self.bounds_x[1]-1
							o = sess.run(
										oracle.oracle,
										feed_dict={
											oracle.inputs: [self.cv_state_onehot[state_index].tolist()]
										})
							zmap[y][x] = o[0][1]
							

				# sc = plt.scatter(x_s, y_s, marker = 's', c = c_s, vmin=np.min(c_s), vmax=np.max(c_s), s= scatter_size, cmap=cm)
				blur_zmap = np.zeros_like(zmap)
				ksize = 5
				for ii in range(blur_zmap.shape[0]):
					for jj in range(blur_zmap.shape[1]):
						neighbors = []
						for ki in range(max(0, ii - ksize), min(blur_zmap.shape[0], ii + ksize)):
							for kj in range(max(0, jj - ksize), min(blur_zmap.shape[1], jj + ksize)):
								if self.MAP[ki][kj] != 'x':
									neighbors.append(zmap[ki][kj])
						blur_zmap[ii][jj] = np.mean(neighbors)

				sc = plt.imshow(blur_zmap, cmap = cm)
				plt.scatter(wx_s, wy_s, marker = 's', c = 'black', s = scatter_size)
				plt.colorbar(sc)
				np.save("zmap_{}_{}".format(j, i), zmap)
		fig = mpl.pyplot.gcf()
		fig.set_size_inches(24, 29)
		fig.savefig('Zmap_blah.png', bbox_inches='tight', dpi = 250)

if __name__ == '__main__':
	ter = Terrain(2)
	ter.plot_z("/home/hmi/Desktop/PROJECT/plot_figure/map2_sharee_only/z_8/1000/")