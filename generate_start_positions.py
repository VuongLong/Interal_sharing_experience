import matplotlib.pyplot as plt
import random
import numpy as np
import json
from env.map import ENV_MAP
from env.sxsy import SXSY
from random import randint

MAP = ENV_MAP[5]['map']
bounds_x = ENV_MAP[5]['size_x']
bounds_y = ENV_MAP[5]['size_y']
SMAP = []
SXSY = {}
for i in range(1000):
	start = []
	for i in range(20):
		sx = 0
		sy = 0
		while MAP[sy][sx]==0:    
			sx = randint(1,bounds_x[1]) 
			sy = randint(1,bounds_y[1]) 
		start.append([sx,sy])
	SMAP.append(start)	

SXSY[5] = SMAP	
file = open('./env/sxsy.py','a')	
file.write('SXSY = ')
file.write(json.dumps(SXSY))

