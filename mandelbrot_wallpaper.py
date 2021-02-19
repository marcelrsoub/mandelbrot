from matplotlib import pyplot as plt
from matplotlib import colors
from numba import jit
import numpy as np

@jit
def mandelbrot(size=1500,limits=0,n=500,style='default'):
	
	width=size
	height=round(width/1.5)

	matrix=np.ones([width, height])

	if limits==0: limits=[-2.0,0.5,-1.25,1.25] #defining limits if not difined


	for x,re in enumerate(np.linspace(limits[0],limits[1],width)):
		for y,im in enumerate(np.linspace(limits[2],limits[3],height)):
			c=complex(re,im)
			z=0.0j
			for i in range(round(n)):
				if style=='default':
					if abs(z) > 16:
						matrix[x,y]=i
						break
					else:
						z=z*z+c
				elif style=='mosaic':
					if abs(z) > 2:
						p=width
						q=height
						x2=np.int(np.floor(p*np.mod(np.real(np.log10(np.log10(z)*q/p)),1)/255.0))
						y2=np.int(np.floor((q*np.mod(2*np.angle(c),1)+1)/255.0))
					
						matrix[x,y]=x2+y2+(i+x2+y2)/(i+1) #style colorful .. with cmap
						break
					else:
						z=z*z+c

				elif style=='thickness':
					if abs(z) > 100:
						if (i%2==0):
							matrix[x,y]=1
						else:
							matrix[x,y]=0
						
						break
					else:
						z=z**2+c
	return matrix



zoom=1 # default: 1
bouge=[
		-0.25	#droite - gauche
		,0		#haut-bas
]


limits=[(-2.0+bouge[0])/zoom,(1+bouge[0])/zoom,(-0.9-bouge[1])/zoom,(0.9-bouge[1])/zoom]
# limits= [-0.7429718683436934, -0.7419430617593313, -0.20634357910818318, -0.20565770805194175]


# -------4K WALLPAPER------------
# pixels=3600
# width=30
# height=round(width/1.5)


pixels=600
width=5
height=round(width/1.5)


fig=plt.figure(dpi=120, frameon=False)
# cmap = 'gnuplot2'
cmap = 'gray'

fig.set_size_inches(width,height)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

matrix=mandelbrot(pixels,limits,style='mosaic',n=100)

ax.imshow(matrix.T, cmap, interpolation="bilinear", extent=limits)
plt.show()
# plt.savefig('wallpaper.png',bbox_inches=None,frameon=None, pad_inches=0)