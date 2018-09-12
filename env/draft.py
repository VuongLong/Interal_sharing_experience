from sxsy import SXSY 
import numpy as np 
import matplotlib.pyplot as plt 

# pts = SXSY[4]
# s = [l for ep in pts for l in ep]
# s = np.array(s).astype(int)
# sx = s[:, 0]
# sy = s[:, 1]

# plt.figure()
# cm = plt.cm.get_cmap('RdYlBu')
# sc = plt.scatter(sx[:10], sy[:10], c = range(10), vmin = 0, vmax = 20, s = 35, cmap = cm)
# plt.colorbar(sc)
# plt.show()

x = np.arange(20000)
y = [0.99 ** np.sqrt(i) for i in x]

plt.plot(x, y)
plt.show()