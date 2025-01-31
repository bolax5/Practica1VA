import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('./training/frontal_2.jpg', 0)          # queryImage
img2 = cv2.imread('./training/frontal_18.jpg', 0)  # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img3 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()