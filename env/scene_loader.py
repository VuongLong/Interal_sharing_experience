# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import os
from .constants import ACTION_SIZE
from .constants import HISTORY_LENGTH

class THORDiscreteEnvironment(object):

  def __init__(self, config=dict()):

    # configurations
    self.scene_name          = config.get('scene_name', 'bedroom_04')
    self.random_start        = config.get('random_start', True)
    self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling
    self.terminal_state_id   = config.get('terminal_state_id', [0])
    self.anti_collision      = config.get('anti_collision', 0)
    self.success_reward      = config.get('success_reward', 10.0)

    self.h5_file_path = config.get('h5_file_path', '{}.h5'.format(self.scene_name))
    self.h5_file      = h5py.File(os.path.join("env", self.h5_file_path), 'r')

    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()]
    self.n_locations = self.locations.shape[0]

    self.terminals = np.zeros(self.n_locations)
    self.terminals[self.terminal_state_id] = 1
    self.terminal_states, = np.where(self.terminals)

    self.cv_action_onehot = np.identity(ACTION_SIZE, dtype=int)

    self.resnet_features = self.h5_file['resnet_feature']

    self.transition_graph = self.h5_file['graph'][()]
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

    self.history_length = HISTORY_LENGTH

    # we use pre-computed fc7 features from ResNet-50
    # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
    self.s_t      = np.zeros([2048, self.history_length])
    self.s_target = self._tiled_state(self.terminal_state_id)

    self.reset()

  # public methods

  def reset(self):
    # randomize initial state
    while True:
      k = random.randrange(self.n_locations)
      min_d = np.inf
      # check if target is reachable
      for t_state in self.terminal_states:
        dist = self.shortest_path_distances[k][t_state]
        min_d = min(min_d, dist)
      # min_d = 0  if k is a terminal state
      # min_d = -1 if no terminal state is reachable from k
      if min_d > 0: break

    # reset parameters
    self.current_state_id = k
    self.s_t = self._tiled_state(self.current_state_id)

    self.reward   = 0
    self.collided = False
    self.terminal = False

  def step(self, action):
    assert not self.terminal, 'step() called in terminal state'
    k = self.current_state_id
    if self.transition_graph[k][action] != -1:
      self.current_state_id = self.transition_graph[k][action]
      if self.terminals[self.current_state_id]:
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

  def _tiled_state(self, state_id):
    k = random.randrange(self.n_feat_per_locaiton)
    f = self.resnet_features[state_id][k][:,np.newaxis]
    return np.tile(f, (1, self.history_length))

  def _reward(self, terminal, collided):
    # positive reward upon task completion
    if terminal: return self.success_reward
    # time penalty or collision penalty
    return -0.1 if collided and self.anti_collision else -0.01

  def state(self, state_id):
    # read from hdf5 cache
    k = random.randrange(self.n_feat_per_locaiton)
    return self.resnet_features[state_id][k]

  # properties
  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE 

  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
    return action_vocab[:ACTION_SIZE]

  @property
  def observation(self):
    return self.h5_file['observation'][self.current_state_id]

  @property
  def target(self):
    return self.terminal_states

  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def z(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]

if __name__ == "__main__":
  scene_name = 'bedroom_04'

  env = THORDiscreteEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })