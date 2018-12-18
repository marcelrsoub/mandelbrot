from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from numba import jit
import numpy as np

@jit
def triangle(size):

	matrix=np.ones([size, size])

	y=size

	while (y>=0):
		i=0
		x=0
		while (x+y < size):
			if ((x&y)!=0):
				matrix[x,y]=1
			else:
				matrix[x,y]=0
			x+=1
		y=y-1
	return matrix

matrix=triangle(1024)

mpl.rcParams['toolbar'] = 'None' #erase buttons

fig=plt.figure(dpi=72,frameon=False)

cmap = 'gray'

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

img=ax.imshow(matrix, cmap, interpolation="bilinear")
plt.show()