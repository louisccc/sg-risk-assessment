import cv2
import os, sys, pdb
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np

H_OFFSET = 370
ORIGINAL_H = 720
ORIGINAL_W = 1280
IMAGE_H = 350
IMAGE_W = 1280
impath = "00017003.jpg"

src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[int(IMAGE_W*16/33), IMAGE_H], [int(IMAGE_W*17/33), IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = cv2.imread(impath, cv2.IMREAD_UNCHANGED) # Read the test img
#img = cv2.resize(img, (IMAGE_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) #resize and interpolate
img = img[H_OFFSET:(H_OFFSET+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()
#pdb.set_trace()
loc = np.array([[[300,175]]], dtype='float32')
loc2 = cv2.perspectiveTransform(loc, M)
print(loc2[0][0]) 
