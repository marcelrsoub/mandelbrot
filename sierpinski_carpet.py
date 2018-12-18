from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from numba import jit
import numpy as np

 
@jit
def carpet(degre):
    size=3**degre
    matrix=np.ones([size, size])

    for niveau in range(degre+1):
        step=3**(degre-niveau)
        for x in range(size):
            if x%3==1:
                for y in range(size):
                    if y%3==1:
                        matrix[y*step:(y+1)*step, x*step:(x+1)*step]=0
                
    return matrix


degre=6
matrix=carpet(degre)
zoom=1 # default: 1
bouge=[
        0   #droite - gauche
        ,0      #haut-bas
]

mpl.rcParams['toolbar'] = 'None' #erase buttons


fig=plt.figure(dpi=72,frameon=False)

cmap = 'gray'

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

img=ax.imshow(matrix, cmap, interpolation="bilinear")



plt.show()