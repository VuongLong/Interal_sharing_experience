import matplotlib.pyplot as plt
import random
from random import randint
import numpy as np
from env.map import ENV_MAP
import json
SXSY = {}

for i in range(3):
	index = i+1
	MAP = ENV_MAP[index]['map']
	bounds_x = ENV_MAP[index]['size_x']
	bounds_y = ENV_MAP[index]['size_y']
	SMAP = []
	for i in range(2000):
		start = []
		for i in range(24):
			sx = 0
			sy = 0
			while MAP[sy][sx]==0:    
				sx = randint(1,bounds_x[1]) 
				sy = randint(1,bounds_y[1]) 
			start.append([sx,sy])
		SMAP.append(start)	

	SXSY[index] = SMAP	
file = open('./env/sxsy.py','w')	
file.write('SXSY = ')
file.write(json.dumps(SXSY))

