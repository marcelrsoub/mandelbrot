# %%
from logging import raiseExceptions
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
        self.mode = 'real_time'  # or 'animation'
        self.style = 'normal'

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

        self.stop_calculation = False  # stop threads
        self.number_of_divisions = 2
        self.mouse_down = False

    # @jit(forceobj=True)
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
        if self.style == 'normal':
            for i in range(int(n)):
                if(self.stop_calculation == True):
                    break
                z = np.power(z, 2)+complex_matrix
                mask = np.abs(z) > self.calculation_limit
                matrix[mask] = i
        elif self.style == 'thickness':
            for i in range(int(n)):
                if(self.stop_calculation == True):
                    break
                z = np.power(z, 2)+complex_matrix
                mask = np.abs(z) > self.calculation_limit
                if mask.any():
                    matrix[mask] = i % 2
        elif self.style == 'mosaic':
            for i in range(int(n)):
                if(self.stop_calculation == True):
                    break
                z = np.power(z, 2)+complex_matrix
                mask = np.abs(z) > self.calculation_limit
                if mask.any():
                    p = width
                    q = height
                    x2 = (
                        np.floor(p*np.mod(np.real(np.log10(np.log10(z)*q/p)), 1)/255.0))
                    y2 = (np.floor((q*np.mod(2*np.angle(complex_matrix), 1)+1)/255.0))
                    # matrix[mask] = x2[mask]+y2[mask]+(i+x2[mask]+y2[mask])/(i+1)
                    matrix[mask] = np.angle(complex_matrix)[
                        mask]+i % 2+y2[mask]

        else:
            raise ValueError("style `"+self.style+"` non existant.")

        return matrix

    def threadSequence(self, limits, size):

        threads = np.array([])

        newlimits = self.split_limits(limits)

        matrix = np.zeros((self.number_of_divisions**2, int(size /
                                                            self.number_of_divisions), int(size/self.number_of_divisions)))

        tic = time.perf_counter()

        for i in range(self.number_of_divisions**2):
            t = threading.Thread(target=self.mandelbrot_core_calculation, args=(
                matrix[i, :, :], newlimits[i]))
            threads = np.append(threads, t)
            t.start()
        while t.is_alive():
            t.join()

        if(self.stop_calculation == True):

            return

        row = np.zeros((self.number_of_divisions, int(
            size/self.number_of_divisions), int(size/self.number_of_divisions)))
        rows = np.zeros((self.number_of_divisions, int(size/self.number_of_divisions),
                         int(size/self.number_of_divisions)*self.number_of_divisions))
        for i in range(self.number_of_divisions):
            for counter, j in enumerate(range(i*self.number_of_divisions, i*self.number_of_divisions+self.number_of_divisions)):
                row[counter] = matrix[j]
            rows[i] = np.hstack(row)
            row = np.zeros((self.number_of_divisions, int(
                size/self.number_of_divisions), int(size/self.number_of_divisions)))

        matrix = np.vstack(rows)

        # print("min:",np.min(matrix))
        # print("max:",np.max(matrix))

        toc = time.perf_counter()
        print(f"Mandelbrot calculated in {toc - tic:0.4f} seconds") # executes on every resolution loop
        self.matrix_full = matrix

        self.update_mandelbrot(matrix)

        return matrix

    def split_limits(self, limits):
        newlimits1 = np.zeros((self.number_of_divisions**2, 4))

        xm = np.abs(limits[1]-limits[0])/(self.number_of_divisions)
        ym = (limits[3]-limits[2])/(self.number_of_divisions)

        x_vector = np.linspace(
            limits[0], limits[1]-xm, self.number_of_divisions)
        y_vector = np.linspace(
            limits[2], limits[3]-ym, self.number_of_divisions)

        newlimits = np.zeros((self.number_of_divisions**2, 4))
        counter = 0
        for i, x in enumerate(x_vector):
            for j, y in enumerate(y_vector):

                limit_here = np.array([x, x+xm, y, y+ym])
                newlimits[counter] = limit_here
                counter += 1

        return newlimits

    def update_mandelbrot(self, matrix):

        if(self.mode == 'real_time'):
            self.img.set_data(matrix.T)
            self.img.set_extent(self.limits)
            self.fig.canvas.draw()
            # self.ax.set_axis_off()  # erasing buttons
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
        if(self.mode == 'refresh_graph'):
            print("Graph refreshed.")
            self.img = self.ax.imshow(matrix.T, self.cmap, extent=self.limits,
                                      interpolation='hanning', norm=colors.Normalize(vmin=0, vmax=self.iterations/2))
            self.fig.canvas.draw()
            self.mode = 'real_time'

    def onclick(self, event):
        self.mouse_down = True;
        self.x0=event.xdata
        self.y0=event.ydata
        x0 = self.limits[0]+abs(self.limits[1]-self.limits[0])/2
        y0 = self.limits[2]+abs(self.limits[3]-self.limits[2])/2

        self.bouge = [0, 0]

        # self.limits = [(self.limits[0]-self.bouge[0]), (self.limits[1]-self.bouge[0]),
        #                (self.limits[2]+self.bouge[1]), (self.limits[3]+self.bouge[1])]

    def onpan(self,event):
        if(self.mouse_down and (event.xdata != None or event.ydata != None)):
            self.stop_calculation = True
            # x0 = self.limits[0]+abs(self.limits[1]-self.limits[0])/2
            # y0 = self.limits[2]+abs(self.limits[3]-self.limits[2])/2

            multiplier=1
            size=20

            self.bouge = [-(self.x0 -event.xdata*multiplier), -(self.y0-event.ydata*multiplier)]

            limits = [(self.limits[0]-self.bouge[0]), (self.limits[1]-self.bouge[0]),
                        (self.limits[2]+self.bouge[1]), (self.limits[3]+self.bouge[1])]
            # self.size=10
            self.stop_calculation=False;
            # self.resolution_loop(self.limits)
            matrix = np.zeros((20, 20))
            self.mandelbrot_core_calculation(
                matrix, limits=limits)  # calculate first mandelbrot
            self.update_mandelbrot(matrix)
            print("x:",event.xdata)
            print("y:",event.ydata)

    def onrelease(self, event):
        self.mouse_down = False;
        self.stop_calculation = True
        # x0 = self.limits[0]+abs(self.limits[1]-self.limits[0])/2
        # y0 = self.limits[2]+abs(self.limits[3]-self.limits[2])/2

        # self.bouge = [x0-event.xdata, y0-event.ydata]

        self.limits = [(self.limits[0]-self.bouge[0]), (self.limits[1]-self.bouge[0]),
                       (self.limits[2]+self.bouge[1]), (self.limits[3]+self.bouge[1])]

        if self.printLimits:
            print("limits=", self.limits)

        limits = self.limits
        self.stop_calculation = False
        self.resolution_loop(limits)

    def resolution_loop(self, limits):

        if(self.mode == 'real_time' or self.mode == 'refresh_graph'):

            for taille in [100, self.size]:
                t = threading.Thread(target=self.threadSequence, args=(
                    self.limits, taille))
                t.start()
            self.stop_calculation = False
        elif(self.mode == 'animation'):
            self.threadSequence(self.limits, self.size)

    def generateZoomAnimation(self, final_limits=[-0.7765779444472669, -0.7765638318740933, -0.13442108343165082, -0.13440697085847714], zoom=0.87, frames=150):
        self.mode = 'animation'
        self.limits = final_limits
        # self.pltInit()
        step_size = int((self.iterations-100)/frames)
        initial_iterations = self.iterations

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

            self.iterations = initial_iterations * \
                np.exp(np.log(100/initial_iterations)/frames*i)
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

    def saveImage(self, width=4000):
        canvas = np.zeros((width, width))
        fig = plt.figure(dpi=120, frameon=False)

        fig.set_size_inches(width, width)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        print("Generating high-res image.")
        matrix = self.mandelbrot_core_calculation(
            canvas, self.limits)

        ax.imshow(matrix.T, self.cmap, extent=self.limits,
                                      interpolation='hanning', norm=colors.Normalize(vmin=0, vmax=self.iterations/2))
        # plt.show()
        plt.savefig('mandelbrot.png',bbox_inches=None,frameon=None, pad_inches=0)
        print("Image saved as 'mandelbrot.png'.")


    def onscroll(self, event):
        self.stop_calculation = True
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
        self.stop_calculation = False
        self.resolution_loop(limits)

    def onpress(self, event):
        # print("key pressed:",event.key)
        if event.key == 'up':
            self.stop_calculation = True
            self.iterations += self.iterations*0.25
            print("iterations=", self.iterations)
            self.mode = 'refresh_graph'
            self.stop_calculation = False
            self.resolution_loop(self.limits)
        elif(event.key == 'down'):
            self.stop_calculation = True
            self.iterations -= self.iterations*0.25
            print("iterations=", self.iterations)
            self.mode = 'refresh_graph'
            self.stop_calculation = False
            self.resolution_loop(self.limits)
        elif(event.key == 'ctrl+s'):
            self.saveImage()
        elif(event.key == 'ctrl+l'):
            print("limits=", self.limits)

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

    def deZoom(self, zoom, frames):
        """Method that enables the user to preview the zooming out executed by generateZoomAnimation

        Args:
            zoom (float): <1 if zooming out and >1 if zooming in
            frames (int): number of frames
        """
        for i in range(frames):
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
            matrix.T, self.cmap, extent=self.limits, norm=colors.Normalize(),interpolation="bilinear")

        self.fig.canvas.mpl_connect(
            'scroll_event', self.onscroll)  # listen to events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onpan)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)

        plt.show()


if __name__ == "__main__":
    newMand = Mandelbrot(500)
    # newMand.limits = [-1.348944719491528, -1.3489445452622297, -0.0628381714457414, -0.06283799721644297]
    # newMand.limits= [-0.7765715290179841, -0.7765715096591733, -0.1344143356761133, -0.13441431631730238]
    # newMand.limits= [-0.7765719564040663, -0.7765703883403803, -0.1344149614496769, -0.13441339338599093]
    # newMand.limits= [-0.8539408213940964, -0.8538138082355331, -0.23550382944673076, -0.23543238454503895]
    # newMand.limits= [-1.37190425354845, -1.3684748982672428, -0.0097858590032773, -0.006356503722070172]
    # newMand.limits= [-0.9171078484470955, -0.9171078462961165, -0.27754717237749293, -0.27754717022651393]
    newMand.iterations = 100
    newMand.calculation_limit = 2
    newMand.number_of_divisions = 2 # higher calculation speed in 2 tested on 8-core cpu's
    # newMand.style='thickness'
    # newMand.generateZoomAnimation(final_limits=newMand.limits, zoom=0.89, frames=150)

    # newMand.deZoom(0.89,149)
    newMand.show()

    # DONE: loop from low res to high res
    # DONE: (PAN FUNCTIONALITY) on click starts calculation in low res, on release launches resolution loop or full res
    # TODO: Pan doesn't recalculate already generated areas
    # TODO: saving 4k wallpaper method

# %%
