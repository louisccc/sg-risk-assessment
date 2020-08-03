import cv2
import os, sys, pdb
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np

CROPPED_H = 390
ORIGINAL_H = 720
ORIGINAL_W = 1280
impath = "35011.jpg"
H_OFFSET = ORIGINAL_H - CROPPED_H
BIRDS_EYE_IMAGE_H = 620
BIRDS_EYE_IMAGE_W = 1280

src = np.float32([[0, CROPPED_H], [ORIGINAL_W, CROPPED_H], [0, 0], [ORIGINAL_W, 0]])
dst = np.float32([[int(BIRDS_EYE_IMAGE_W*16/33), BIRDS_EYE_IMAGE_H], [int(BIRDS_EYE_IMAGE_W*17/33), BIRDS_EYE_IMAGE_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = cv2.imread(impath, cv2.IMREAD_UNCHANGED) # Read the test img
#img = cv2.resize(img, (IMAGE_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) #resize and interpolate
img = img[H_OFFSET:ORIGINAL_H, 0:ORIGINAL_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (BIRDS_EYE_IMAGE_W, BIRDS_EYE_IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
#pdb.set_trace()
loc = np.array([[[300,100]]], dtype='float32')
loc2 = cv2.perspectiveTransform(loc, M)
print(loc2[0][0]) 
#plt.plot(loc2[0][0][0], loc2[0][0][1], color='cyan', marker='o')
plt.show()
