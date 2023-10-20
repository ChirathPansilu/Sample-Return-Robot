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

image = mpimg.imread('sample.jpg')

warped = perspect_transform(image, source, destination)
colorsel = color_thresh(warped, rgb_thresh=(160, 160, 160))

plt.imshow(colorsel, cmap='gray')
plt.show()

ypos, xpos = colorsel.nonzero()
plt.plot(xpos, ypos, '.')
plt.xlim(0, 320)
plt.ylim(0, 160)
plt.show() 

def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    
    y_pixel = (xpos - binary_img.shape[1]/2).astype(np.float64)
    x_pixel = (binary_img.shape[0] - ypos).astype(np.float64)

    return x_pixel, y_pixel

nx, ny = rover_coords(colorsel)
plt.plot(nx, ny, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
plt.show() 

def rotate_pix(xpix, ypix, yaw):
    yaw_rad = yaw * np.pi/180

    x_rotated = xpix*np.cos(yaw_rad) - ypix*np.sin(yaw_rad)
    y_rotated = xpix*np.sin(yaw_rad) + ypix*np.cos(yaw_rad)

    return x_rotated, y_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    x_translated = np.int_(xpos + (xpix_rot/scale))
    y_translated = np.int_(ypos + (ypix_rot/scale))
    
    return x_translated, y_translated

def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    
    # Clipping to world map size
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)

    return x_pix_world, y_pix_world

# mock values
rover_yaw = np.random.random(1)*360

rover_xpos = np.random.random(1)*160 + 20
rover_ypos = np.random.random(1)*160 + 20

# rover coords
xpix, ypix = rover_coords(colorsel)

# 200 x 200 worldmap
worldmap = np.zeros((200, 200))

# scale factor of 10 between world space pixels and rover space pixels
scale = 10

# world coords
x_world, y_world = pix_to_world(xpix, 
                                ypix, 
                                rover_xpos, 
                                rover_ypos, 
                                rover_yaw, 
                                worldmap.shape[0], 
                                scale)

# Add pixel to worldmap
worldmap[y_world, x_world] += 1

print('Xpos =', rover_xpos, 'Ypos =', rover_ypos, 'Yaw =', rover_yaw)

# rover centric coords
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
f.tight_layout()
ax1.plot(xpix, ypix, '.')
ax1.set_title('Rover Space', fontsize=40)
ax1.set_ylim(-160, 160)
ax1.set_xlim(0, 160)
ax1.tick_params(labelsize=20)

# world coords
ax2.imshow(worldmap, cmap='gray')
ax2.set_title('World Space', fontsize=40)
ax2.set_ylim(0, 200)
ax2.tick_params(labelsize=20)
ax2.set_xlim(0, 200)
plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
plt.show()