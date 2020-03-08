from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import time
# from numba import jit
import numpy as np
import threading
# import multiprocessing as mp


class Mandelbrot:

	def __init__(self,resolution=16/9,size=1000):
		super().__init__()

		self.size=1000 #maximum size in pixels. The highest the more calculation power it takes.
		self.calculation_limit=4

		self.zoom=1 # initializing current zoom value. default: 1
		self.row=[0,0,0,0]

		self.bouge=[
				0	#droite - gauche
				,0		#haut-bas
		]

		#resolution in proportion width/height
		self.resolution=resolution

		#invert resolution to have it in percentage
		resolution=1./resolution

		#define height to be applied
		resolution_height=resolution*2.5

		#limits coordinates
		self.limits=[
			(-2.+self.bouge[0])/self.zoom,
			(0.5+self.bouge[0])/self.zoom,
			(-resolution_height/2-self.bouge[1])/self.zoom,
			(resolution_height/2-self.bouge[1])/self.zoom
		]

		
	

	# @jit
	def mandelbrot_core_calculation(self, size, limits, threading_index, doThread=True): #main method

		width=np.int_(size)
		height=np.int_(width*self.resolution)

		matrix=np.zeros([width, height])

		dx=(limits[1]-limits[0])
		# print("n:",n)
		# print("dx:",dx)

		for x,re in enumerate(np.linspace(limits[0],limits[1],width)):
			for y,im in enumerate(np.linspace(limits[2],limits[3],height)):
				c=complex(re,im)
				z=0.0j
				n=1.0917*(limits[1]-limits[0])**(-0.068) #resolution factor
				for i in range(int(n*100)):
					
					if abs(z)**2 > self.calculation_limit:
						matrix[x,y]=i
						break
					else:
						# z=z*z+(c**3-1)/(2*c+1)  #mandelbrot + newton fractals
						z=z*z+c
		if doThread:
			self.row[threading_index]=matrix
		else:
			self.current_mandelbrot=matrix
			return matrix


	def thread_mandelbrot(self,size,limits):
		self.mandelbrot_core_calculation(size,limits)

	def threadSequence(self,limits):
		threads=list()
		newlimits=[0,0,0,0]

		# for i in range(1,np.int_(self.size/50)):
			# t=mp.Process(target=self.thread_mandelbrot, args=(i*200,limits), daemon=True)
			# # threads.append(t)
			# t.start()
			# print(i*50)

		xm=(limits[1]-limits[0])/2 #=1.25
		ym=(limits[3]-limits[2])/2 #=0.7

		leftcorner=[limits[0],limits[2]]
		
		newlimits[0]=[leftcorner[0],leftcorner[0]+xm,leftcorner[1],leftcorner[1]+ym]
		newlimits[1]=[leftcorner[0]+xm,leftcorner[0]+2*xm,leftcorner[1],leftcorner[1]+ym]
		newlimits[2]=[leftcorner[0],leftcorner[0]+xm,leftcorner[1]+ym,leftcorner[1]+2*ym]
		newlimits[3]=[leftcorner[0]+xm,leftcorner[0]+2*xm,leftcorner[1]+ym,leftcorner[1]+2*ym]

		for i in range(4):
			t=threading.Thread(target=self.mandelbrot_core_calculation, args=(self.size/4,newlimits[i],i))
			# threads.append(t)
			t.start()
			t.join()
		row1=np.vstack((self.row[0],self.row[1]))
		row2=np.vstack((self.row[2],self.row[3]))
		self.current_mandelbrot=np.hstack((row1,row2))
		self.recalculate_mandelbrot(limits)

	def recalculate_mandelbrot(self,limits):
		
		matrix=self.current_mandelbrot
		self.img.set_data(matrix.T)
		self.img.set_extent(self.limits)
		self.fig.canvas.draw()
		self.fig.clf()
		self.ax.set_axis_off() #erasing buttons
		self.fig.add_axes(self.ax)
		self.img=self.ax.imshow(matrix.T, self.cmap, interpolation="bilinear", extent=limits)
		plt.show()

	def onclick(self, event):

		x0=self.limits[0]+abs(self.limits[1]-self.limits[0])/2
		y0=self.limits[2]+abs(self.limits[3]-self.limits[2])/2

		self.bouge=[x0-event.xdata,y0-event.ydata]

		self.limits=[(self.limits[0]-self.bouge[0]),(self.limits[1]-self.bouge[0]),(self.limits[2]+self.bouge[1]),(self.limits[3]+self.bouge[1])]

		print("limits=", self.limits)

		limits=self.limits
		self.threadSequence(limits)

	def onscroll(self, event):

		if event.step>0:
			self.zoom = 3*event.step
			print("zoom:",zoom)
		else: 
			self.zoom = 0.7/abs(event.step)
			print("zoom:",zoom)
		
		limits=self.limits

		self.limits=[
				limits[0]*(1+1/zoom)/2+limits[1]*(1-1/zoom)/2,  #calculate new x0
				limits[1]*(1+1/zoom)/2+limits[0]*(1-1/zoom)/2,	#calculate new x1
				limits[2]*(1+1/zoom)/2+limits[3]*(1-1/zoom)/2,	#calculate new y0
				limits[3]*(1+1/zoom)/2+limits[2]*(1-1/zoom)/2	#calculate new y1
		]

		print("limits=", self.limits)
		limits=self.limits
		self.threadSequence(limits)

	def show(self):

		matrix=self.mandelbrot_core_calculation(size=self.size,limits=self.limits,threading_index=0, doThread=False) #calculate first mandelbrot 


		mpl.rcParams['toolbar'] = 'None' #erase buttons

		self.fig=plt.figure(dpi=120,frameon=False)
		self.cmap = 'gnuplot2'

		thismanager = plt.get_current_fig_manager()
		# thismanager.window.wm_iconbitmap("mandel.ico")
		thismanager.set_window_title('Mandelbrot Set')

		self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
		self.ax.set_axis_off()
		self.fig.add_axes(self.ax)

		self.img=self.ax.imshow(matrix.T, self.cmap, interpolation="bilinear", extent=self.limits)

		self.fig.canvas.mpl_connect('scroll_event', self.onscroll)  #listen to events
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)

		plt.show()
		# self.threadSequence(self.limits)


if __name__ == "__main__":
	newMand=Mandelbrot()
	newMand.size=200
	newMand.show()