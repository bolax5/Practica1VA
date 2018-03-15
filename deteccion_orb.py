import cv2
import numpy as np
matrix= []
for x in range (48):
    I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
    matrix.append(I)

orb = cv2.ORB_create(nfeatures=100, nlevels=4)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6, # 12
                     key_size = 12,     # 20
                     multi_probe_level = 1) #2
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

keyPoints = []
descriptores = []
for img in range (48):
    pts1 = orb.detect(matrix[img], None)
    pts1, des1 = orb.compute(matrix[img],pts1)
    descriptores.append(des1)
    keyPoints.append(pts1)

flann.add(descriptores)


imgTest = cv2.imread("./testing/test1.jpg", 0)
pts2 = orb.detect(imgTest,None)
pts2, des2 = orb.compute(imgTest,pts2)

zipped=zip(des2, pts2)
for (d,kp) in zipped:
    lp= flann.knnMatch(d,k=2)
    lp
