# %%
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import time
from numba import jit
import numpy as np
import threading

from numpy.core.numeric import Inf


class Mandelbrot:

    def __init__(self, size, resolution=1/1):
        super().__init__()

        # maximum size in pixels. The highest the more calculation power it takes.
        self.size = size
        self.calculation_limit = 2

        self.zoom = 1  #  initializing current zoom value. default: 1

        self.printLimits = True
        self.mode = 'real_time'

        self.bouge = [
            0  # droite - gauche
            , 0  # haut-bas
        ]

        # resolution in proportion height/width
        self.resolution = 1./resolution
        self.width = int(self.size)
        self.height = int(self.width*self.resolution)

        self.row = np.zeros((4, int(self.width/2), int(self.width/2)))

        self.iterations = 1E2

        # define height to be applied
        resolution_height = resolution*2.5  # FIXME:

        # limits coordinates
        self.limits = [
            (-2.+self.bouge[0])/self.zoom,
            (0.5+self.bouge[0])/self.zoom,
            (-resolution_height/2-self.bouge[1])/self.zoom,
            (resolution_height/2-self.bouge[1])/self.zoom
        ]

    # @jit
    def mandelbrot_core_calculation(self, canvas, limits):  # main method

        width = int(len(canvas))
        height = int(width)

        matrix = canvas

        # dx=(limits[1]-limits[0])

        # print("dx:",dx)
        x_value_vec = np.linspace(limits[0], limits[1], width)
        y_value_vec = np.linspace(limits[2], limits[3], height)

        complex_matrix = np.broadcast_to(
            x_value_vec, (width, width))+y_value_vec.reshape(width, 1)*1j
        complex_matrix = complex_matrix.T
        z = complex_matrix
        # n=1/(abs(limits[1]-limits[0])*2)*10 #resolution factor
        n = self.iterations
        # n=int(3.0917E2*(limits[1]-limits[0])**(-0.2)) #resolution factor
        # print("n:",n)
        for i in range(int(n)):
            z = np.power(z,2,dtype=np.complex64)+complex_matrix
            mask = np.abs(z,dtype=np.complex64) > self.calculation_limit
            matrix[mask] = i

    def threadSequence(self, limits, size):
        threads = np.array([])

        newlimits = self.split_limits(limits)

        matrix = np.zeros((4, int(size/2), int(size/2)))

        tic = time.perf_counter()

        for i in range(4):
            t = threading.Thread(target=self.mandelbrot_core_calculation, args=(
                matrix[i, :, :], newlimits[i]))
            threads = np.append(threads, t,dtype=np.complex64)
            t.start()
        while t.is_alive():
            t.join()

        row1 = np.vstack((matrix[0], matrix[1]))
        row2 = np.vstack((matrix[2], matrix[3]))

        matrix = np.hstack((row1, row2))
        # print("min:",np.min(matrix))
        # print("max:",np.max(matrix))

        toc = time.perf_counter()
        print(f"Mandelbrot calculated in {toc - tic:0.4f} seconds")

        return matrix

    def split_limits(self, limits):
        newlimits = np.zeros((4, 4))

        xm = (limits[1]-limits[0])/2
        ym = (limits[3]-limits[2])/2

        leftcorner = [limits[0], limits[2]]

        newlimits[0] = [leftcorner[0], leftcorner[0] +
                        xm, leftcorner[1], leftcorner[1]+ym]
        newlimits[1] = [leftcorner[0]+xm, leftcorner[0] +
                        2*xm, leftcorner[1], leftcorner[1]+ym]
        newlimits[2] = [leftcorner[0], leftcorner[0] +
                        xm, leftcorner[1]+ym, leftcorner[1]+2*ym]
        newlimits[3] = [leftcorner[0]+xm, leftcorner[0] +
                        2*xm, leftcorner[1]+ym, leftcorner[1]+2*ym]

        return newlimits

    def update_mandelbrot(self, matrix):

        if(self.mode == 'real_time'):
            self.img.set_data(matrix.T)
            self.img.set_extent(self.limits)
            self.fig.canvas.draw()
            self.ax.set_axis_off()  # erasing buttons
        if(self.mode == 'animation'):
            self.fig = plt.figure(frameon=False)
            self.cmap = 'gnuplot2'
            thismanager = plt.get_current_fig_manager()
            thismanager.set_window_title('Mandelbrot Set')

            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

            print(self.iterations)
            self.img = self.ax.imshow(matrix.T, self.cmap, extent=self.limits,
                                      interpolation='hanning', norm=colors.Normalize(vmin=0, vmax=self.iterations/2))

    def onclick(self, event):

        x0 = self.limits[0]+abs(self.limits[1]-self.limits[0])/2
        y0 = self.limits[2]+abs(self.limits[3]-self.limits[2])/2

        self.bouge = [x0-event.xdata, y0-event.ydata]

        self.limits = [(self.limits[0]-self.bouge[0]), (self.limits[1]-self.bouge[0]),
                       (self.limits[2]+self.bouge[1]), (self.limits[3]+self.bouge[1])]

        if self.printLimits:
            print("limits=", self.limits)

        limits = self.limits
        self.resolution_loop(limits)

    def resolution_loop(self, limits):
        # TODO for taille in [300,500,1000]:

        matrix = self.threadSequence(self.limits, self.size)
        self.update_mandelbrot(matrix)

    def generateZoomAnimation(self, final_limits=[-0.7765779444472669, -0.7765638318740933, -0.13442108343165082, -0.13440697085847714], zoom=0.87, frames=150):
        self.mode = 'animation'
        self.limits = final_limits
        # self.pltInit()
        step_size = int((self.iterations-100)/frames)

        for i in range(frames):
            print(str((i+1)/frames*100)+' %', flush=True)
            # print("zoom:",zoom)

            limits = self.limits

            limits = [
                limits[0]*(1+1/zoom)/2+limits[1] *
                (1-1/zoom)/2,  # calculate new x0
                limits[1]*(1+1/zoom)/2+limits[0] * \
                (1-1/zoom)/2,  # calculate new x1
                limits[2]*(1+1/zoom)/2+limits[3] * \
                (1-1/zoom)/2,  # calculate new y0
                limits[3]*(1+1/zoom)/2+limits[2] * \
                (1-1/zoom)/2  # calculate new y1
            ]

            # if self.printLimits:
            # 	print("limits=", limits)

            self.limits = limits
            # self.pltInit()
            self.resolution_loop(limits)

            self.iterations -= step_size
            if(len(str(i)) == 1):
                filename = "000"+str(i)
            elif((len(str(i)) == 2)):
                filename = "00"+str(i)
            elif((len(str(i)) == 3)):
                filename = "0"+str(i)
            else:
                filename = str(i)
            plt.savefig('./anim/'+filename+'.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close()

    def onscroll(self, event):

        if event.step > 0:
            zoom = 3*event.step
            print("zoom:", zoom)
        else:
            zoom = 0.7/abs(event.step)
            print("zoom:", zoom)

        limits = self.limits

        self.limits = [
            limits[0]*(1+1/zoom)/2+limits[1]*(1-1/zoom)/2,  # calculate new x0
            limits[1]*(1+1/zoom)/2+limits[0]*(1-1/zoom)/2,  # calculate new x1
            limits[2]*(1+1/zoom)/2+limits[3]*(1-1/zoom)/2,  # calculate new y0
            limits[3]*(1+1/zoom)/2+limits[2]*(1-1/zoom)/2  # calculate new y1
        ]

        if self.printLimits:
            print("limits=", self.limits)
        limits = self.limits
        self.resolution_loop(limits)

    def pltInit(self):
        matrix = np.zeros((self.size, self.size))
        self.mandelbrot_core_calculation(
            matrix, limits=self.limits)  # calculate first mandelbrot
        mpl.rcParams['toolbar'] = 'None'  # erase buttons

        self.fig = plt.figure(frameon=False)
        self.cmap = 'gnuplot2'
        thismanager = plt.get_current_fig_manager()
        # thismanager.window.wm_iconbitmap("./mandel.ico") #FIXME: icon doesn't work
        thismanager.set_window_title('Mandelbrot Set')

        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)

        self.img = self.ax.imshow(
            matrix.T, self.cmap, extent=self.limits, norm=colors.Normalize(), aspect='equal')

    def deZoom(self, zoom, times):
        for i in range(times):
            limits = self.limits

            self.limits = [
                limits[0]*(1+1/zoom)/2+limits[1] *
                (1-1/zoom)/2,  # calculate new x0
                limits[1]*(1+1/zoom)/2+limits[0] * \
                (1-1/zoom)/2,  # calculate new x1
                limits[2]*(1+1/zoom)/2+limits[3] * \
                (1-1/zoom)/2,  # calculate new y0
                limits[3]*(1+1/zoom)/2+limits[2] * \
                (1-1/zoom)/2  # calculate new y1
            ]

    def show(self):

        matrix = np.zeros((self.size, self.size))
        self.mandelbrot_core_calculation(
            matrix, limits=self.limits)  # calculate first mandelbrot

        mpl.rcParams['toolbar'] = 'None'  # erase buttons

        self.fig = plt.figure(frameon=False)
        self.cmap = 'gnuplot2'

        thismanager = plt.get_current_fig_manager()
        # thismanager.window.wm_iconbitmap("./mandel.ico") #FIXME: icon doesn't work
        thismanager.set_window_title('Mandelbrot Set')

        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)

        self.img = self.ax.imshow(
            matrix.T, self.cmap, extent=self.limits, norm=colors.Normalize())

        self.fig.canvas.mpl_connect(
            'scroll_event', self.onscroll)  # listen to events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()


if __name__ == "__main__":
    newMand = Mandelbrot(800)
    newMand.limits = [-1.348944719491528, -1.3489445452622297, -
                      0.0628381714457414, -0.06283799721644297]
    # newMand.limits= [-0.7765715290179841, -0.7765715096591733, -0.1344143356761133, -0.13441431631730238]
    # newMand.limits= [-0.7765719564040663, -0.7765703883403803, -0.1344149614496769, -0.13441339338599093]
    # newMand.limits= [-0.8539408213940964, -0.8538138082355331, -0.23550382944673076, -0.23543238454503895]
    # newMand.limits= [-1.37190425354845, -1.3684748982672428, -0.0097858590032773, -0.006356503722070172]
    # newMand.limits= [-0.9171078484470955, -0.9171078462961165, -0.27754717237749293, -0.27754717022651393]
    newMand.iterations = 13000
    newMand.calculation_limit = 2
    newMand.generateZoomAnimation(final_limits=newMand.limits, zoom=0.89, frames=150)

    # newMand.deZoom(0.89,149)
    # newMand.show()

    # TODO: loop from low res to high res
    # TODO: saving 4k wallpaper method

# %%
