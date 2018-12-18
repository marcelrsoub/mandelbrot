from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from numba import jit
import numpy as np


@jit
def mandelbrot(size=1500,limits=0): #main function
	
	width=size
	height=width

	matrix=np.zeros([width, height])

	if limits==0: limits=[-2.0,0.5,-1.25,1.25] #define limits if not defined

	dx=(limits[1]-limits[0])
	# print("n:",n)
	# print("dx:",dx)

	for x,re in enumerate(np.linspace(limits[0],limits[1],width)):
		for y,im in enumerate(np.linspace(limits[2],limits[3],height)):
			c=complex(re,im)
			z=0.0j
			n=1.0917*(limits[1]-limits[0])**(-0.068) #resolution factor
			for i in range(round(n*100)):
				
				if abs(z) > 10:
					matrix[x,y]=i
					break
				else:
					# z=z*z+(c**3-1)/(2*c+1)  #mandelbrot + newton fractals
					z=z*z+c
	return matrix



zoom=1 # default: 1
bouge=[
		0	#droite - gauche
		,0		#haut-bas
]
# limits=[(-2.0+bouge[0])/zoom,(0.5+bouge[0])/zoom,(-1.25-bouge[1])/zoom,(1.25-bouge[1])/zoom]  #limits for 1:1
limits=[(-2.0+bouge[0])/zoom,(1.+bouge[0])/zoom,(-1.-bouge[1])/zoom,(1.-bouge[1])/zoom]  #limits for widescreen 3:2

matrix=mandelbrot(1000,limits) #calculate first mandelbrot

mpl.rcParams['toolbar'] = 'None' #erase buttons

fig=plt.figure(dpi=120,frameon=False)
cmap = 'gnuplot2'

thismanager = plt.get_current_fig_manager()
thismanager.window.wm_iconbitmap("mandel.ico")
thismanager.set_window_title('Mandelbrot Set')

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

img=ax.imshow(matrix.T, cmap, interpolation="bilinear", extent=limits)


def onclick(event):

    global bouge, limits, zoom
    x0=limits[0]+abs(limits[1]-limits[0])/2
    y0=limits[2]+abs(limits[3]-limits[2])/2
    bouge=[x0-event.xdata,y0-event.ydata]
    
    
    limits=[(limits[0]-bouge[0]),(limits[1]-bouge[0]),(limits[2]+bouge[1]),(limits[3]+bouge[1])]

    print("limits=", limits)

    size=100 
    while size<=1500: #systeme de resolution interactive
	    matrix=mandelbrot(size, limits)
	    img.set_data(matrix.T)
	    img.set_extent(limits)
	    fig.canvas.draw()
	    size+=700

def onscroll(event):

	global bouge, limits, zoom

	if event.step>0:
		zoom = 3*event.step
		print("zoom:",zoom)

		
	else: 
		zoom = 0.7/abs(event.step)
		print("zoom:",zoom)

	limits=[
			limits[0]*(1+1/zoom)/2+limits[1]*(1-1/zoom)/2,  #calculate new x0
			limits[1]*(1+1/zoom)/2+limits[0]*(1-1/zoom)/2,	#calculate new x1
			limits[2]*(1+1/zoom)/2+limits[3]*(1-1/zoom)/2,	#calculate new y0
			limits[3]*(1+1/zoom)/2+limits[2]*(1-1/zoom)/2	#calculate new y1
	]

	print("limits=", limits)

	global fig,ax,cmap,img
	size=100 
	while size<=1500: # real time calculus
	    matrix=mandelbrot(size, limits)
	    img.set_data(matrix.T)
	    img.set_extent(limits)
	    fig.canvas.draw()
	    size+=700
	fig.clf()
	ax.set_axis_off() #erasing buttons
	fig.add_axes(ax)
	img=ax.imshow(matrix.T, cmap, interpolation="bilinear", extent=limits)
	plt.show()

fig.canvas.mpl_connect('scroll_event', onscroll)  #listen to events
fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()

