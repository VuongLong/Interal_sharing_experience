import gym
import gym_dang
import pygame
from utilise import *


def main():
	env = gym.make('dang-v1')
	config_value = getValueConfig()
	env.config(config_value)
	print('Running gym example')
	observation = env.reset()
	print(observation.shape)

	while 1:
		actions = env.get_action_list()
		try:
			mode = input('Input:')
			if mode == 'r':
				observation = env.reset()
				continue
			elif mode == 'q':
				break
			elif mode == 't':
				pos = input("position:")
				x, y, z = pos.split(',')
			else:
				mode = int(mode) - 1
				mode = len(actions)-1 if mode >= len(actions) else mode
		except ValueError:
			print("Not a number")
			mode = 0

		if mode != 't':
			observation, reward, done = env.step(mode)
		else:
			observation, reward, done = env.teleport(float(x), float(y), int(z))

		print(reward, env.position)
		if reward > 1: 
			print("done")
			break



if __name__ == "__main__":
    main()
