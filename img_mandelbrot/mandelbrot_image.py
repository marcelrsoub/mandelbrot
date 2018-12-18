import scipy.misc as misc
import imageio
from matplotlib import pyplot as plt
from matplotlib import colors
from numba import jit
import numpy as np

@jit
def mandelbrot(size=1500,limits=0,n=100,img=misc.face()):
	
	width=size
	height=size
	# height=round(width/1.5)

	matrix=np.ndarray(shape=(width,height,3),dtype=int)

	if limits==0: limits=[-2.0,0.5,-1.25,1.25] #defining limits if not defined

	p=len(img[:,1])
	q=len(img[1,:])

	for x,re in enumerate(np.linspace(limits[0],limits[1],width)):
		for y,im in enumerate(np.linspace(limits[2],limits[3],height)):
			c=complex(re,im)
			z=0.0j

			for i in range(round(n)):

				z=z*z+c

				if abs(z) > 25:
					x2=np.int(np.floor(p*np.mod(np.real(np.log(np.log(z)*q/p)),1)/2))
					y2=np.int(np.floor((q*np.mod(2*np.angle(c),1)+2)/2))
					
					# matrix[x,y]=x2+y2+(i+x2+y2)/(i+1) #style colorful .. with cmap
					color=img[x2,y2]

					matrix[x,y]=color
					
					break

	return matrix



zoom=1 # default: 1
bouge=[
		-0.25	#droite - gauche
		,0		#haut-bas
]


limits=[(-2.0+bouge[0])/zoom,(1+bouge[0])/zoom,(-1.5-bouge[1])/zoom,(1.5-bouge[1])/zoom]
# limits= [-0.7429718683436934, -0.7419430617593313, -0.20634357910818318, -0.20565770805194175]


# -------4K WALLPAPER------------
pixels=3600
width=30
height=30

pixels=2400
width=10
height=width


pixels=500
width=5
height=5


fig=plt.figure(dpi=120, frameon=False)

fig.set_size_inches(width,height)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

img=imageio.imread('new.jpg')

matrix=mandelbrot(size=pixels,n=100,img=img)

ax.imshow(matrix,interpolation='bilinear', extent=limits)
# plt.imshow()

plt.savefig('wallpaper.png',bbox_inches=None,frameon=None, pad_inches=0)
plt.show()
