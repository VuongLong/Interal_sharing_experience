import gym
import gym_dang

import numpy as np
import cv2
import h5py
import json
from utilise import *

def dump(scene, resolution=(300, 300)):

	f = h5py.File("../env/dumped/{}.hdf5".format(scene), "w")

	observations = []
	locations = []
	visible_objects = []

	env = gym.make('dang-v1')
	config_value = getValueConfig()
	env.config(config_value)
	env.reset()

	print('Running gym example')
	world_map = env.worldMap

	locations = []
	for i in range(len(world_map)):
		for j in range(len(world_map[0])):
			if world_map[i][j] == 0:
				locations.append((i + 0.5, j + 0.5))

	states = []

	for loc in locations:
		for rot in [0, 90, 180, 270]:
			states.append((loc[0], loc[1], rot))

	loc2idx = dict(zip(states, range(len(states))))
	graph = np.zeros(shape=(len(states), 4), dtype=int)

	directions = {0: -1, 90: 1, 180: 1, 270: -1}

	for state in states:
		loc = (state[0], state[1])
		rot = state[2]

		to_states = []

		if rot == 0 or rot == 180:
			to_states.append((loc[0] + directions[rot], loc[1], rot)) # move ahead
			to_states.append((loc[0] - directions[rot], loc[1], rot)) # move back

		else:
			to_states.append((loc[0], loc[1] + directions[rot], rot)) # move ahead
			to_states.append((loc[0], loc[1] - directions[rot], rot)) # move back

		to_states.append((loc[0], loc[1], rot - 90 if rot >= 90 else 270)) # turn left
		to_states.append((loc[0], loc[1], rot + 90 if rot <= 180 else 0)) # turn right
		
		state_idx = loc2idx[state]
		for i, new_state in enumerate(to_states):
			if new_state in loc2idx:
				graph[state_idx][i] = loc2idx[new_state]
			else:
				graph[state_idx][i] = -1

	for state in states:
		observation, _, _ = env.teleport(state[0], state[1], int(state[2]))

		resized_frame = cv2.resize(observation, (resolution[0], resolution[1]))
		observations.append(resized_frame)

	print("{} states".format(len(states)))

	f.create_dataset("locations", data=np.asarray(states, np.float32))
	f.create_dataset("observations", data=np.asarray(observations, np.uint8))
	f.create_dataset("goals", data=np.array([[1.5, 6.5], [7.5, 1.5]], dtype=np.float32))
	f.create_dataset("graph", data=graph)
	f.close()

if __name__ == '__main__':
	
	scene = "Pygame_10x10"
	dump(scene, (128, 128))
