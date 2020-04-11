#%%
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import time
from numba import jit
import numpy as np
import threading


class Mandelbrot:

	def __init__(self,size,resolution=1/1):
		super().__init__()

		self.size=size #maximum size in pixels. The highest the more calculation power it takes.
		self.calculation_limit=2

		self.zoom=1 # initializing current zoom value. default: 1
		
		
		self.printLimits=True

		self.bouge=[
				0	#droite - gauche
				,0		#haut-bas
		]

		#resolution in proportion height/width
		self.resolution=1./resolution
		self.width=np.int_(self.size)
		self.height=np.int_(self.width*self.resolution)

		self.row=np.zeros((4,np.int_(self.width/2),np.int_(self.width/2)))

		self.iterations=1E2

		#define height to be applied
		resolution_height=resolution*2.5 #FIXME:

		#limits coordinates
		self.limits=[
			(-2.+self.bouge[0])/self.zoom,
			(0.5+self.bouge[0])/self.zoom,
			(-resolution_height/2-self.bouge[1])/self.zoom,
			(resolution_height/2-self.bouge[1])/self.zoom
		]

		
	

	# @jit
	def mandelbrot_core_calculation(self,canvas, limits): #main method

		width=np.int(len(canvas))
		height=np.int(width)

		matrix=canvas

		# dx=(limits[1]-limits[0])
		# print("n:",n)
		# print("dx:",dx)
		x_value_vec=np.linspace(limits[0],limits[1],width)
		y_value_vec=np.linspace(limits[2],limits[3],height)

		complex_matrix=np.broadcast_to(x_value_vec,(width,width))+y_value_vec.reshape(width,1)*1j
		complex_matrix=complex_matrix.T
		z=complex_matrix
		# n=1.0917*(limits[1]-limits[0])**(-0.068) #resolution factor
		for i in range(int(self.iterations)):
			z=z*z+complex_matrix
			mask=abs(z) > self.calculation_limit
			matrix[mask]=i


	def threadSequence(self,limits,size):
		threads=np.array([])
		newlimits=np.zeros((4,4))

		xm=(limits[1]-limits[0])/2 #=1.25
		ym=(limits[3]-limits[2])/2 #=0.7

		leftcorner=[limits[0],limits[2]]
		
		newlimits[0]=[leftcorner[0],leftcorner[0]+xm,leftcorner[1],leftcorner[1]+ym]
		newlimits[1]=[leftcorner[0]+xm,leftcorner[0]+2*xm,leftcorner[1],leftcorner[1]+ym]
		newlimits[2]=[leftcorner[0],leftcorner[0]+xm,leftcorner[1]+ym,leftcorner[1]+2*ym]
		newlimits[3]=[leftcorner[0]+xm,leftcorner[0]+2*xm,leftcorner[1]+ym,leftcorner[1]+2*ym]

		# newlimits=
		
		matrix1=np.zeros((np.int(size/2), np.int(size/2)))
		matrix2=np.zeros((np.int(size/2), np.int(size/2)))
		matrix3=np.zeros((np.int(size/2), np.int(size/2)))
		matrix4=np.zeros((np.int(size/2), np.int(size/2)))

		tic = time.perf_counter()

		for i,matrix in enumerate([matrix1,matrix2,matrix3,matrix4]):
			t=threading.Thread(target=self.mandelbrot_core_calculation, args=(matrix,newlimits[i]))
			threads=np.append(threads,t)
			t.start()
			while t.is_alive():
				t.join()
		row1=np.vstack((matrix1,matrix2))
		row2=np.vstack((matrix3,matrix4))

		toc=time.perf_counter()
		print(f"Mandelbrot calculated in {toc - tic:0.4f} seconds")
		
		return np.hstack((row1,row2))

	def update_mandelbrot(self,matrix,limits):
		self.img.set_data(matrix.T)
		self.img.set_extent(self.limits)
		self.fig.canvas.draw()
		self.ax.set_axis_off() #erasing buttons

	def onclick(self, event):

		x0=self.limits[0]+abs(self.limits[1]-self.limits[0])/2
		y0=self.limits[2]+abs(self.limits[3]-self.limits[2])/2

		self.bouge=[x0-event.xdata,y0-event.ydata]

		self.limits=[(self.limits[0]-self.bouge[0]),(self.limits[1]-self.bouge[0]),(self.limits[2]+self.bouge[1]),(self.limits[3]+self.bouge[1])]

		if self.printLimits:
			print("limits=", self.limits)

		limits=self.limits
		self.resolution_loop(limits)

	def resolution_loop(self,limits):
		# for i in np.linspace(100.,self.size,np.int_(self.size/200)):
		for i in [self.size]:
			matrix=self.threadSequence(self.limits,i)
			self.update_mandelbrot(matrix,limits)
			

	def onscroll(self, event):

		if event.step>0:
			zoom = 3*event.step
			print("zoom:",zoom)
		else: 
			zoom = 0.7/abs(event.step)
			print("zoom:",zoom)
		
		limits=self.limits

		self.limits=[
				limits[0]*(1+1/zoom)/2+limits[1]*(1-1/zoom)/2,  #calculate new x0
				limits[1]*(1+1/zoom)/2+limits[0]*(1-1/zoom)/2,	#calculate new x1
				limits[2]*(1+1/zoom)/2+limits[3]*(1-1/zoom)/2,	#calculate new y0
				limits[3]*(1+1/zoom)/2+limits[2]*(1-1/zoom)/2	#calculate new y1
		]

		if self.printLimits:
			print("limits=", self.limits)
		limits=self.limits
		self.resolution_loop(limits)


	def show(self):

		matrix=np.zeros((300,300))
		self.mandelbrot_core_calculation(matrix,limits=self.limits) #calculate first mandelbrot 


		mpl.rcParams['toolbar'] = 'None' #erase buttons

		self.fig=plt.figure(frameon=False)
		self.cmap = 'gnuplot2'

		thismanager = plt.get_current_fig_manager()
		# thismanager.window.wm_iconbitmap("./mandel.ico") #FIXME: icon doesn't work
		thismanager.set_window_title('Mandelbrot Set')

		self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
		self.ax.set_axis_off()
		self.fig.add_axes(self.ax)

		self.img=self.ax.imshow(matrix.T, self.cmap, interpolation="none", extent=self.limits)

		self.fig.canvas.mpl_connect('scroll_event', self.onscroll)  #listen to events
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		
		plt.show()


if __name__ == "__main__":
	newMand=Mandelbrot(300,resolution=1./1.)
	# newMand.limits= [-0.8539408213940964, -0.8538138082355331, -0.23550382944673076, -0.23543238454503895]
	# newMand.limits= [-1.37190425354845, -1.3684748982672428, -0.0097858590032773, -0.006356503722070172]
	# newMand.size=300
	newMand.iterations=100
	newMand.show()

	#TODO: timing between calculations
		#TODO: if time > 3 sec decrease resolution and iterations
	#TODO: loop from low res to high res
	#TODO: saving 4k wallpaper method

# %%
