import os
import sys
import math
from random import randint

import torch
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

from distral.network import (DQN, select_action, optimize_model, 
                            Tensor, optimize_policy, PolicyNetwork)
from distral.memory_replay import ReplayMemory, Transition
from env.terrain import Terrain
from env.sxsy import SXSY

def training(map_index, num_task, logdir, batch_size=128, gamma=0.999, alpha=0.9,
            beta=5, eps_start=0.9, eps_end=0.05, eps_decay=5,
            is_plot=False, num_episodes=200, max_num_ep_per_episode=1000,
            learning_rate=0.001, memory_replay_size=10000,
            memory_policy_size=1000):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    tb = SummaryWriter(logdir)
    
    init_env = Terrain(map_index)
    num_actions = init_env.action_size
    num_envs = num_task
    state_size = init_env.size_m
    list_of_envs = [Terrain(map_index) for task in range(num_task)]

    policy = PolicyNetwork(num_actions, state_size)
    models = [DQN(num_actions, state_size) for _ in range(0, num_envs)]   ### Add torch.nn.ModuleList (?)
    memories = [ReplayMemory(memory_replay_size, memory_policy_size) for _ in range(0, num_envs)]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        policy.cuda()
        for model in models:
            model.cuda()

    optimizers = [optim.Adam(model.parameters(), lr=learning_rate)
                    for model in models]
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    episode_durations = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]

    steps_done = np.zeros(num_envs)
    episodes_done = np.zeros(num_envs)
    current_time = np.zeros(num_envs)
    episode_total_reward = np.zeros(num_envs)

    for task in range(num_task):
        env = list_of_envs[task]
        sx = randint(1, env.bounds_x[1])
        sy = randint(1, env.bounds_y[1])
        list_of_envs[task].resetgame(task, sx, sy)

    while np.min(episodes_done) < num_episodes:
        for task, env in enumerate(list_of_envs):
            # print(task)
            state = env.player.getposition()
            state_index = state[0]+(state[1]-1)*env.bounds_x[1]-1
            state = env.cv_state_onehot[state_index]
            state = Tensor([state])

            action = select_action(state, policy, models[task], num_actions,
                                eps_start, eps_end, eps_decay,
                                episodes_done[task], alpha, beta)
            steps_done[task] += 1
            current_time[task] += 1
            try:
                reward, done = env.player.action(action)
            except:
                print(action)
                raise
            episode_total_reward[task] += reward
            reward = Tensor([reward])

            if not done:
                next_state = env.player.getposition()
                state_index = next_state[0]+(next_state[1]-1)*env.bounds_x[1]-1
                next_state = env.cv_state_onehot[state_index]
                next_state = Tensor([next_state])
            else:
                next_state = None

            time = Tensor([current_time[task]])
            memories[task].push(state, action, next_state, reward, time)

            optimize_model(policy, models[task], optimizers[task],
                            memories[task], batch_size, alpha, beta, gamma)

            if done:
                print("ENV:", task, "iter:", episodes_done[task],
                    "\treward:", episode_total_reward[task],
                    "\tit:", current_time[task], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[task] / eps_decay))
                sx = randint(1, env.bounds_x[1])
                sy = randint(1, env.bounds_y[1])
                env.resetgame(task, sx, sy)
                episodes_done[task] += 1
                episode_durations[task].append(current_time[task])
                episode_rewards[task].append(episode_total_reward[task])
                tb.add_scalar('task_{}/rewards'.format(task), episode_total_reward[task], np.sum(episode_durations[task]))

                current_time[task] = 0
                episode_total_reward[task] = 0

        optimize_policy(policy, policy_optimizer, memories, batch_size,
                num_envs, gamma)

    print('Complete')

if __name__ == '__main__':
    training(1, 2, 'output/log')