# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import os
from .constants import ACTION_SIZE
from .constants import HISTORY_LENGTH

class PyGameDumpEnv(object):

  def __init__(self, config=dict()):

    # configurations
    self.scene_name          = config.get('scene_name', 'Pygame_10x10')
    self.anti_collision      = config.get('anti_collision', 0)
    self.task                = config.get('task', 0)
    self.success_reward      = config.get('success_reward', 10.0)

    self.h5_file_path = config.get('h5_file_path', '{}.hdf5'.format(self.scene_name))
    self.h5_file      = h5py.File(os.path.join("env/dumped/", self.h5_file_path), 'r')

    self.locations = self.h5_file['locations'][()]
    self.observations = self.h5_file['observations'][()].astype(np.float32) / 255.
    self.transition_graph = self.h5_file['graph'][()]
    self.goals   = self.h5_file['goals'][()]

    self.n_locations = self.locations.shape[0]
    self.cv_action_onehot = np.identity(ACTION_SIZE, dtype=int)

    self.history_length = HISTORY_LENGTH
    self.resolution = self.observations[0].shape
    self.history_states = np.zeros([self.history_length, self.resolution[0], self.resolution[1], self.resolution[2]])

    self.MAP = [
            [5,4,4,4,2,2,2,2,4,4],
            [2,9,9,9,9,9,9,4,15,4],
            [5,9,9,9,9,3,3,15,15,4],
            [2,9,2,15,2,3,3,3,15,4],
            [5,9,2,15,2,15,15,15,15,4],
            [2,9,2,15,2,15,15,15,15,4],
            [5,9,2,15,2,15,5,15,2,2],
            [2,9,2,15,15,15,5,15,15,2],
            [5,9,2,15,15,15,5,15,15,2],
            [2,2,2,2,2,2,2,2,2,2]]
            
    self.reset()

  def reset(self):
    # randomize initial state
    # while True:
    #   k = random.randrange(self.n_locations)
    #   # check if target is reachable
    #   dist = [self.shortest_path_distances[k][t_state] for t_state in self.terminal_states]
    #   if dist[0] < self.training_area and dist[0] > 0:
    #     break
    k = random.randrange(self.n_locations)
    # reset parameters
    self.current_state_id = k
    self.update_states()

    self.reward   = 0
    self.collided = False
    self.terminal = False

  def step(self, action):
    assert not self.terminal, 'step() called in terminal state'
    k = self.current_state_id
    if self.transition_graph[k][action] != -1:
      self.current_state_id = self.transition_graph[k][action]
      if np.all(self.locations[self.current_state_id] == self.goals[self.task]):
        self.terminal = True
        self.collided = False
      else:
        self.terminal = False
        self.collided = False
    else:
      self.terminal = False
      self.collided = True

    self.reward = self._reward(self.terminal, self.collided)

  # private methods

  def update_states(self):
    o = self.observations[self.current_state_id]
    self.history_states = np.append(self.history_states[1:, :], np.expand_dims(o, 0), 0)

  def _reward(self, terminal, collided):
    # positive reward upon task completion
    if terminal: return self.success_reward
    # time penalty or collision penalty
    return -0.1 if collided and self.anti_collision else -0.01

  def state(self, state_id):
    return self.observations[state_id]

  # properties
  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE 

  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "MoveBackward", "RotateLeft", "RotateRight"]
    return action_vocab[:ACTION_SIZE]

  @property
  def observation(self):
    return self.observations[self.current_state_id]

  @property
  def target(self):
    return self.goals

if __name__ == "__main__":
  scene_name = 'bedroom_04'

  env = THORDiscreteEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })