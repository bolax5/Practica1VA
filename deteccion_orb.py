import cv2
import numpy as np
matrix= []
for x in range (48):
    I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
    matrix.append(I)

orb = cv2.ORB_create(nfeatures=100, nlevels=4)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

keyPoints = []
descriptores = []
for img in range (48):
    pts1 = orb.detect(matrix[img], None)
    pts1, des1 = orb.compute(matrix[img],pts1)
    des1 = np.float32(des1)
    descriptores.append(des1)
    keyPoints.append(pts1)

imgTest = cv2.imread("./testing/test1.jpg", 0)
pts2 = orb.detect(imgTest,None)
pts2, des2 = orb.compute(imgTest,pts2)
des2 = np.float32(des2)

flann_params = dict(algorithm=1, trees=4)
indices = []
nCercanos = 3
for i in range(1,48):
    flannIndex = cv2.flann_Index(features=descriptores[i], params=flann_params, distType=None)
    idx, dist = flannIndex.knnSearch(des2, nCercanos, params={})
    indices.append(idx)
votaciones = [0 for i in range (100)]
votaciones = np.uint32(votaciones)
indices = np.uint32(indices)
for i in range(1,47):
    for j in range (1, nCercanos+1):
        votaciones[indices[i][j]] = votaciones[indices[i][j]] +1
votaciones