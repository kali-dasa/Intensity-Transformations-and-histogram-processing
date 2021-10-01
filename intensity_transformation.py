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

"""2. Point/Intensity transformation: Log transformation"""

from PIL import Image
import numpy as np
import matplotlib.pylab as plt
im_g = im.convert('L')
plt.imshow(im_g)

im_g=np.array(im_g)

c = 255/(np.log(1 + np.max(im_g))) 
log_transformed = c * np.log(1 + im_g) 
  
log_transformed = np.array(log_transformed, dtype = np.uint8) 
  
plt.title("log transformation")
plt.imshow(log_transformed, cmap=plt.cm.gray)
