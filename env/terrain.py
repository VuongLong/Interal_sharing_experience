import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
from random import randint
try:
    from .sxsy import SXSY
    from .map import ENV_MAP
    from .controller import Player
except ModuleNotFoundError:
    from sxsy import SXSY
    from map import ENV_MAP
    from controller import Player

class Terrain:
    def __init__(self, map_index):
        self.reward_locs = ENV_MAP[map_index]['goal']
        self.MAP = ENV_MAP[map_index]['map']
        self.bounds_x = ENV_MAP[map_index]['size_x']
        self.bounds_y = ENV_MAP[map_index]['size_y']

        self.action_size = 8
        self.reward_range = 1.0
        self.reward_goal = 1.0
        
        self.num_task = len(self.reward_locs)

        self.cv_state_onehot = np.identity(self.bounds_x[1]*self.bounds_y[1],dtype=int)
        self.cv_action_onehot = np.identity(self.action_size,dtype=int)
        self.cv_task_onehot = np.identity(len(self.reward_locs),dtype=int)
        


    def getreward(self):
        done = False
        reward = -0.05

        x_pos, y_pos = self.reward_locs[self.task]
        
        if self.MAP[self.player.y][self.player.x] == 0:
            reward = -1.0 
            done = True
            return reward, done

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
            plt.scatter([x_pos,], [y_pos,], marker='^', color="red")

        plt.xlim([self.bounds_x[0]-1,self.bounds_x[1]+1])
        plt.ylim([self.bounds_y[0]-1,self.bounds_y[1]+1])

        for y in range(self.bounds_y[0]-1,self.bounds_y[1]+2):
            for x in range(self.bounds_x[0]-1,self.bounds_x[1]+2):
                if self.MAP[y][x]==0:
                    plt.scatter([x,], [y,], marker='o', color="green")

        count = np.load('count.npy')
        for i in range(count.shape[0]):
            for j in range(count.shape[1]):
                if count[i][j] > 0:
                    plt.scatter(i, j, marker='x', color="red")
                    plt.annotate(count[i][j], (i, j))

        # for (x, y) in SXSY[4][1]:
        #     plt.scatter(x, y, marker='x', color="red")

        # plt.scatter([self.player.x,], [self.player.y,], marker='x', color="red")
        # plt.pause(0.001)
        plt.show()

    def resetgame(self, task, sx, sy):
        #self.player = Player(7, 1, self)
       
        self.player = Player(sx, sy, self)

        self.task = task
            
if __name__ == '__main__':
    ter = Terrain(4)
    ter.resetgame(1, 1, 1)
    ter.plotgame()