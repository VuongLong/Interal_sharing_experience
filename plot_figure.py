import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
import os

class PlotFigure(object):
	
	def __init__(self, save_name, env):
		self.env = env
		self.range_x = [0, self.env.bounds_x[1]+1]
		self.range_y = [0, self.env.bounds_y[1]+1]
		self.save_name = save_name

	def _plot_point(self, ax, point, angle, length):
		x, y = point

		endy = length * math.sin(math.radians(angle)) + y
		endx = length * math.cos(math.radians(angle)) + x

		ax.plot([x, endx], [y, endy])

	def _plot_star(self, ax, orig, lengths, max_length=0.5, angles=[270,225,180,135,90,45,0,315]):
		max_len = max(lengths)
		for i, angle in enumerate(angles):
			self._plot_point(ax, orig, angle, lengths[i]*1.0 / max_len * max_length)

	def plot(self, policy, epoch):
		plt.clf()
		
		for index in range(self.env.num_task):
			ax = plt.subplot(1,2,index+1)
			plt.title(str(epoch))
			for x in range(self.range_x[0]+1,self.range_x[1]):
				for y in range(self.range_y[0]+1,self.range_y[1]):
					if self.env.MAP[y][x]!=0:
						if self.env.action_size == 8:
							self._plot_star(ax, (x, y), policy[x,y,index])
						else:	
							self._plot_star(ax, (x, y), [policy[x,y,index][0],0,policy[x,y,index][1],0,policy[x,y,index][2],0,policy[x,y,index][3],0])
					if self.env.MAP[y][x] ==2:
						plt.plot([x,], [y,], marker='o', markersize=2, color="red")
					else:	
						plt.plot([x,], [y,], marker='o', markersize=2, color="green")
		
		if not os.path.exists('../plot/'+self.save_name):
			os.makedirs('../plot/'+self.save_name)
		plt.savefig('../plot/'+self.save_name+'/'+str(epoch)+'.png', bbox_inches='tight')
		#plt.pause(0.00001)