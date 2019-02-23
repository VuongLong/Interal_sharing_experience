import h5py
import random
import numpy as np

h5_file = h5py.File("dumped/Pygame_10x10.hdf5", 'r')

locations = h5_file['locations'][()]

n_locations = locations.shape[0]

starts = [[], []]
for i in range(2):
      for j in range(15000):
            temp = list(range(24))
            np.random.shuffle(temp)
            starts[i].append(temp)

with open("starts.py", "w") as f:
      f.write("sxsy = " + str(starts))