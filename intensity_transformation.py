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

"""3. Histogram Processing

from PIL import Image
import numpy as np
import matplotlib.pylab as plt
im = Image.open("/content/parrot.jpg") # RGB input image
im_r, im_g, im_b = im.split() # To split the RGB image into 3 channels
plt.style.use('ggplot')
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.imshow(im, cmap='gray')
plt.title('original image', size=20)
plt.axis('off')
plt.subplot(122)
# Use the function here to plot histogram of im_r
# Use the function here to plot histogram of im_g
# Use the function here to plot histogram of im_b
img = cv2.imread("/content/parrot.jpg")
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.xlabel('pixel value', size=20)
plt.ylabel('frequency', size=20)
plt.title('histogram for RGB channels', size=20)
plt.show()

from PIL import Image
import numpy as np
import matplotlib.pylab as plt
im = Image.open("/content/parrot.jpg") # RGB input image
im_r, im_g, im_b = im.split() # To split the RGB image into 3 channels

plt.imshow(im_r)

plt.imshow(im_g)

plt.imshow(im_b)

""" 3.1) Histogram Equalization """
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import equalize_hist
img = cv2.imread("/content/parrot.jpg")
plt.imshow(img)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

eq = np.float32(equalize_hist(img))
plt.imshow(eq)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([eq],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

