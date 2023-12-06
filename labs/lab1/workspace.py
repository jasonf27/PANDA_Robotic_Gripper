from math import pi
import numpy as np

from lib.calculateFK import FK
from core.interfaces import ArmController

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
 ]

# TODO: create plot(s) which visualize the reachable workspace of the Panda arm,
# accounting for the joint limits.
#
# We've included some very basic plotting commands below, but you can find
# more functionality at https://matplotlib.org/stable/index.html

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# TODO: update this with real results
counter = 0
for a in range(-28973, 28973, 15000):
	for b in range(-17628,17628,15000):
		for c in range(-28973,28973,15000):
			for d in range(-30718, -698,15000):
				for e in range(-28973,28973,15000):
					for f in range(-175,37525,15000):
						for g in range(-28973,28973,15000):
							q = np.array([a,b,c,d,e,f,g])
							q = np.true_divide(q,10000)
							joints, t0e = fk.forward(q)
							counter = counter + 1
							print(counter)
							ax.scatter(t0e[0,3],t0e[1,3],t0e[2,3])
	

plt.show()
