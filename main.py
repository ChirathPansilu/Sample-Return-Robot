# Import libraries
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read image 
filename = "sample.jpg"
image = mpimg.imread(filename)
plt.imshow(image)
plt.show()

# Image details
print(image.dtype, image.shape, np.min(image), np.max(image))

# Each channels
red_channel = np.copy(image)
red_channel[:,:,[1,2]] = 0

green_channel = np.copy(image)
green_channel[:,:,[0,2]] = 0

blue_channel = np.copy(image)
blue_channel[:,:,[0,1]] = 0

fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(red_channel)
plt.subplot(132)
plt.imshow(green_channel)
plt.subplot(133)
plt.imshow(blue_channel)
plt.show()