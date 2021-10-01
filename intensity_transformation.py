1. Point/Intensity transformation: Negative of an image.
First print the maximum intensity value of input image and then subtract the intensity values
from max value
"""

from PIL import Image
import numpy as np
import matplotlib.pylab as plt

im = Image.open("/content/parrot.jpg")

plt.imshow(im)

max=0
im.size

largest = np.amax(im)
print(largest)

"""Negative"""

for i in range(0, im.size[0] - 1): 
    for j in range(0, im.size[1] - 1): 
        pixel = im.getpixel((i,j))
        pixel1 = 255 - pixel[0] 
        pixel2 = 255 - pixel[1] 
        pixel3 = 255 - pixel[2] 
        im.putpixel( (i,j),(pixel1,pixel2,pixel3) )

plt.imshow(im) 
plt.show()
