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

        self.zoom=1 # initializing current zoom value. default: 1

        self.printLimits=True

        self.bouge=[
                0    #droite - gauche
                ,0        #haut-bas
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

        # print("dx:",dx)
        x_value_vec=np.linspace(limits[0],limits[1],width)
        y_value_vec=np.linspace(limits[2],limits[3],height)

        complex_matrix=np.broadcast_to(x_value_vec,(width,width))+y_value_vec.reshape(width,1)*1j
        complex_matrix=complex_matrix.T
        z=complex_matrix
        # n=1/(abs(limits[1]-limits[0])*2)*10 #resolution factor
        # print("n:",n)
        n=self.iterations
        for i in range(int(n)):
            z=z*z+complex_matrix
            mask=abs(z) > self.calculation_limit
            matrix[mask]=i

    def threadSequence(self,limits,size):
        threads=np.array([])


        newlimits= self.split_limits(limits)

        matrix=np.zeros((4,np.int(size/2), np.int(size/2)))

        tic = time.perf_counter()

        for i in range(4):
            t=threading.Thread(target=self.mandelbrot_core_calculation, args=(matrix[i,:,:],newlimits[i]))
            threads=np.append(threads,t)
            t.start()
        while t.is_alive():
            t.join()

        row1=np.vstack((matrix[0],matrix[1]))
        row2=np.vstack((matrix[2],matrix[3]))

        matrix=np.hstack((row1,row2))

        toc=time.perf_counter()
        print(f"Mandelbrot calculated in {toc - tic:0.4f} seconds")

        return matrix

    def split_limits(self,limits):
        newlimits=np.zeros((4,4))

        xm=(limits[1]-limits[0])/2
        ym=(limits[3]-limits[2])/2

        leftcorner=[limits[0],limits[2]]

        newlimits[0]=[leftcorner[0],leftcorner[0]+xm,leftcorner[1],leftcorner[1]+ym]
        newlimits[1]=[leftcorner[0]+xm,leftcorner[0]+2*xm,leftcorner[1],leftcorner[1]+ym]
        newlimits[2]=[leftcorner[0],leftcorner[0]+xm,leftcorner[1]+ym,leftcorner[1]+2*ym]
        newlimits[3]=[leftcorner[0]+xm,leftcorner[0]+2*xm,leftcorner[1]+ym,leftcorner[1]+2*ym]

        return newlimits

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
        # for taille in [300,500,1000]:
        for taille in [self.size]:

            matrix=self.threadSequence(self.limits,taille)
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
                limits[1]*(1+1/zoom)/2+limits[0]*(1-1/zoom)/2,    #calculate new x1
                limits[2]*(1+1/zoom)/2+limits[3]*(1-1/zoom)/2,    #calculate new y0
                limits[3]*(1+1/zoom)/2+limits[2]*(1-1/zoom)/2    #calculate new y1
        ]

        if self.printLimits:
            print("limits=", self.limits)
        limits=self.limits
        self.resolution_loop(limits)

    def show(self):

        matrix=np.zeros((self.size,self.size))
        self.mandelbrot_core_calculation(matrix,limits=self.limits) #calculate first mandelbrot

        mpl.rcParams['toolbar'] = 'None' #erase buttons

        self.fig=plt.figure(frameon=False)
        self.cmap = 'plasma'
#         self.cmap = 'viridis'

        thismanager = plt.get_current_fig_manager()
        # thismanager.window.wm_iconbitmap("./mandel.ico") #FIXME: icon doesn't work
        thismanager.set_window_title('Mandelbrot Set')

        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)

        self.img=self.ax.imshow(matrix.T, self.cmap, extent=self.limits)
#         self.img=self.ax.pcolormesh(matrix.T)#, self.cmap, extent=self.limits)

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)  #listen to events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # plt.savefig("divergence=%i.jpg" % (self.iterations))
        plt.savefig("zoom=%e.jpg" % (self.zoom))

        # plt.show()
        plt.close()

    def loop_iteration(self, loop_list):
        for i in loop_list:
            self.iterations = i
            self.show()

    def loop_zoom(self, zoom_list, pos_end=None):
        if pos_end is None:
            pos_end = self.limits

        for z in zoom_list:
            zoom = z
            limits = pos_end
            self.limits = pos_end
            self.limits=[
                    limits[0]*(1+1/zoom)/2+limits[1]*(1-1/zoom)/2,  #calculate new x0
                    limits[1]*(1+1/zoom)/2+limits[0]*(1-1/zoom)/2,    #calculate new x1
                    limits[2]*(1+1/zoom)/2+limits[3]*(1-1/zoom)/2,    #calculate new y0
                    limits[3]*(1+1/zoom)/2+limits[2]*(1-1/zoom)/2    #calculate new y1
            ]

            if self.printLimits:
                print("limits=", self.limits)
            limits=self.limits
            self.resolution_loop(limits)
            self.zoom = zoom
            self.show()


if __name__ == "__main__":
    newMand=Mandelbrot(1000, resolution=1./1.)
    # newMand.iterations=i*100
    # plt.ion()
    plt.ioff()
    # newMand.limits= [-0.8539408213940964, -0.8538138082355331, -0.23550382944673076, -0.23543238454503895]
    # newMand.limits= [-1.37190425354845, -1.3684748982672428, -0.0097858590032773, -0.006356503722070172]
#     newMand.limits= [-0.9171078484470955, -0.9171078462961165, -0.27754717237749293, -0.27754717022651393]
    # newMand.size=300
    # newMand.calculation_limit=16
#     newMand.iterations=2
#     newMand.calculation_limit = -10
    newMand.show()
    # loop_list =range(2, 61)
    # loop_list = [
    #    2,4,10,20,
    # 100]
    # newMand.loop_iteration(loop_list)
    # OPTIMIZE could determine the correct last zoom out to have the full mendelbrot
    zoom = 1/3**13
    pos_end = [-0.10482694888363314, -0.10482538081994718, -0.9283572478726074, -0.9283556798089214]
    zoom=0.0008
    pos_end = [-1.4070716163694568, -1.4036422610882495, -0.001700388660265129, 0.0017289666209419992]
    zoom=0.0000015
    pos_end = [-0.7253085997233288, -0.725303830996902, -0.28783075142042236, -0.287825982693969]

    zoom_list = 10**np.linspace(np.log10(1), np.log10(zoom), 200)
    # zoom_list = np.append(zoom_list , 10**np.linspace(np.log10(zoom/2), np.log10(zoom), 30))
    newMand.loop_zoom(zoom_list, pos_end)


    #TODO: loop from low res to high res
    #TODO: saving 4k wallpaper method

# %%
