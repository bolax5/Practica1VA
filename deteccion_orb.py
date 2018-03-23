import cv2
import numpy as np
import math as mt
matrix= []
centros = []



def votefunc(centro, kparecido, kact):

    "En base a dos puntos de interes asociados, vota en unas coordenadas de la imagen destino"
    #vector = (vector[0], vector[1])
    vector = (centro[0] - kparecido.pt[0], centro[1] - kparecido.pt[1])
    vector = (vector[0]*kact.size/kparecido.size, vector[1]*kact.size/kparecido.size)
    angulo = mt.atan2(vector[1], vector[0])
    angulo = np.rad2deg(angulo)
    angulo = angulo + (kact.angle - kparecido.angle)
    modulo = mt.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    vector = (modulo * mt.cos(angulo) + kact.pt[0], modulo * mt.sin(angulo) + kact.pt[1])

    vector = (np.int(vector[0]), np.int(vector[1]))
    return vector


for x in range (49):
    I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
    matrix.append(I)

orb = cv2.ORB_create(nfeatures=500, nlevels=6)

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

for x in range (1,34):
    imgTest = cv2.imread("./testing/test" + str(x) + ".jpg", 0)

    pts2 = orb.detect(imgTest,None)
    pts2, des2 = orb.compute(imgTest, pts2)
    votaciones = np.zeros((np.int(imgTest.shape[0]/10), np.int(imgTest.shape[1]/10)), dtype=int)
    zipped = zip(des2, pts2)
    for (d, kp) in zipped:
        lp = flann.knnMatch(d, k=6)
        for list in lp:
            for n_kp in list:
                vector = votefunc((225, 110), keyPoints[n_kp.imgIdx][n_kp.trainIdx], kp)
                vector = (np.int(vector[0] / 10), np.int(vector[1] / 10))
                if (vector[0] >= 0) & (vector[1] >= 0) & (vector[0] < (imgTest.shape[0] / 10 -1)) & (vector[1] < (imgTest.shape[1] / 10 -1)):
                    # if vector[0] >= 0 & vector[1] >= 0 & vector[0] < np.int(imgTest.shape[0]/10) & vector[1] < np.int(imgTest.shape[1]/10):

                    votaciones[vector[0]][vector[1]] += 1
    coords = np.unravel_index(votaciones.argmax(), votaciones.shape)
    for i in range(imgTest.shape[0]):
        for j in range(imgTest.shape[1]):
            imgTest[coords[1]*10][j] = 0
            imgTest[coords[1]*10 - 10][j] = 0
            imgTest[i][coords[0]*10] = 0
            imgTest[i][coords[0]*10 - 10] = 0
    cv2.imshow(str(x), imgTest)
    cv2.waitKey()
print(votaciones)