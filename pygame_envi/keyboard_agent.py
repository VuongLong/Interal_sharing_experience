import ai2thor.controller
import sys
import numpy as np
import h5py
import click
import pyglet
import cv2

from PIL import Image

ALL_POSSIBLE_ACTIONS = [
	'MoveAhead',
	'MoveBack',
	'RotateRight',
	'RotateLeft',
	# 'Stop'   
]

class SimpleImageViewer(object):

  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

def key_press(key, mod):
	global human_agent_action, human_wants_restart, stop_requested
	if key == ord('R') or key == ord('r'): # r/R
		human_wants_restart = True
	if key == ord('Q') or key == ord('q'): # q/Q
		stop_requested = True

	if key == 0xFF52: # move ahead
		human_agent_action = 0
	if key == 0xFF54: # move back
		human_agent_action = 1
	if key == 0xFF51: # turn left
		human_agent_action = 2
	if key == 0xFF53: # turn right
		human_agent_action = 3

def run():
	f = h5py.File('Pygame_10x10.hdf5', "r")
	observations = f['observations']
	locations = f['locations']
	graph = f['graph']

	current_position = 115
	print("current position", locations[current_position], graph[current_position])
	for i in range(4):
		if graph[current_position][i] != -1:
			print(i, locations[graph[current_position][i]])
		else:
			print(i, 'collision')

	while True:  # making a loop
		try:  # used try so that if user pressed other than the given key error will not be shown
			key = click.getchar()
			if key =='a':  # Rotate Left
				action = 2
			elif key =='d':
				action = 3
			elif key =='w':
				action = 0
			elif key =='s':
				action = 1
			elif key =='q':
				break
			else:
				print("Key not supported! Try a, d, w, s, q.")

			next_position = int(graph[current_position][action])
			current_position = next_position if next_position != -1 else current_position

			print("current position", locations[current_position], graph[current_position])
			for i in range(4):
				if graph[current_position][i] != -1:
					print(i, locations[graph[current_position][i]])
				else:
					print(i, 'collision')

			cv2.imshow("env", observations[current_position])
			cv2.waitKey(1)

		except:
			print("Key not supported! Try a, d, w, s, q.")

if __name__ == '__main__':
	
	# run()

	human_agent_action = None
	human_wants_restart = False
	stop_requested = False
	next_position = None
	visible = None

	f = h5py.File('../env/dumped/Pygame_10x10.hdf5', "r")
	observations = f['observations']
	locations = f['locations']
	graph = f['graph']
	goals = f['goals']

	# current_position = np.random.randint(0, observations.shape[0])
	current_position = 115
	print(human_agent_action, "current position", locations[current_position], graph[current_position])
	for i in range(4):
		if graph[current_position][i] != -1:
			print(i, locations[graph[current_position][i]])
		else:
			print(i, 'collision')

	viewer = SimpleImageViewer()
	viewer.imshow(observations[current_position].astype(np.uint8))
	viewer.window.on_key_press = key_press

	print("Use arrow keys to move the agent.")
	print("Press R to reset agent\'s location.")
	print("Press Q to quit.")

	while True:
		# waiting for keyboard input
		if human_agent_action is not None:

			# move actions
			next_position = int(graph[current_position][human_agent_action])
			current_position = next_position if next_position != -1 else current_position
			if np.all(locations[current_position][:2] == goals[0]) or np.all(locations[current_position][:2] == goals[1]):
				print('Done')

			human_agent_action = None
			
			print(human_agent_action, "current position", locations[current_position], graph[current_position])
			for i in range(4):
				if graph[current_position][i] != -1:
					print(i, locations[graph[current_position][i]])
				else:
					print(i, 'collision')


		# waiting for reset command
		if human_wants_restart:
			# reset agent to random location
			current_position = np.random.randint(0, observations.shape[0])
			human_wants_restart = False

		# check collision
		if next_position == -1:
			print('Collision occurs.')

		# check quit command
		if stop_requested: break

		viewer.imshow(observations[current_position].astype(np.uint8))

	print("Goodbye.")