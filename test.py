import cv2
import numpy as np

matrix= []
keyPoints = []
descriptores = []

def readTrainingImages():
    for x in range(49):
        I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
        matrix.append(I)

def detectAndComputeTrainingKeypoints():
    for img in range(48):
        pts1 = orb.detect(matrix[img], None)
        pts1, des1 = orb.compute(matrix[img], pts1)
        descriptores.append(des1)
        keyPoints.append(pts1)

        im = cv2.drawKeypoints(matrix[img+1], pts1, None,color=(0,255,0), flags=0)
        cv2.imshow("",im)
        cv2.waitKey()

orb = cv2.ORB_create(nfeatures=500, nlevels=6)

readTrainingImages()
detectAndComputeTrainingKeypoints()