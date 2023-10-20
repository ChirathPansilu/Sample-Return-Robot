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

# Define a function to perform a color threshold
def color_thresh(img, rgb_thresh=(0, 0, 0)):
    color_select = np.zeros_like(img[:,:,0])
    
    above_thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] > rgb_thresh[1]) & (img[:,:,2] > rgb_thresh[2])
    
    color_select[above_thresh] = 1
    
    return color_select
    
    
red_threshold = 160
green_threshold = 160
blue_threshold = 160

rgb_threshold = (red_threshold, green_threshold, blue_threshold)

colorsel = color_thresh(image, rgb_thresh=rgb_threshold)

# Display the original image and binary               
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(colorsel, cmap='gray')
ax2.set_title('Your Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

image = mpimg.imread('example_grid1.jpg')
plt.imshow(image)
plt.show() 

import cv2

def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped

# Source and Destination points
source = np.float32([[14, 140], [118, 95], [200, 95], [300, 140]])

dst_size = 10
offset = 5

destination = np.float32([[image.shape[1]/2 - dst_size/2, image.shape[0] - offset],
                          [image.shape[1]/2 - dst_size/2, image.shape[0] - offset - dst_size], 
                          [image.shape[1]/2 + dst_size/2, image.shape[0] - offset - dst_size],
                          [image.shape[1]/2 + dst_size/2, image.shape[0] - offset]]
                        )      

warped = perspect_transform(image, source, destination)
plt.imshow(warped)
plt.show() 