# To be executed at images folder
import imageio
from os import walk
images = []

_, _, filenames = next(walk("./anim"))

for filename in sorted(filenames):
    # print(filename[0:-3])
    if(filename[-3:]=="png"):
        images.append(imageio.imread("./anim/"+filename))
imageio.mimsave('./movie10.gif', images)