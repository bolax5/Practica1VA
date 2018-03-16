import cv2
import numpy as np
matrix= []
centros = []
keyPoints = []
descriptores = []

def votefunc(centro, kparecido, kact):

    "En base a dos puntos de interes asociados, vota en unas coordenadas de la imagen destino"
    vector = (np.int(centro[0] - kparecido.pt[0]), np.int(centro[1] - kparecido.pt[1]))
    vector = (np.int(vector[0]*(kact.size/kparecido.size) + kact.pt[0]), np.int(vector[1]*(kact.size/kparecido.size) + kact.pt[1]))
    return vector


for x in range (49):
    I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
    matrix.append(I)

orb = cv2.ORB_create(nfeatures=100, nlevels=5)
FLANN_INDEX_LSH = 6

index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6, # 12
                     key_size = 12,     # 20
                     multi_probe_level = 1) #2

search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

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
    votaciones = np.zeros((np.int(imgTest.shape[0]/5), np.int(imgTest.shape[1]/5)), dtype=int)
    zipped = zip(des2, pts2)

    for (d, kp) in zipped:
        lp = flann.knnMatch(d, k=10)

        for list in lp:
            for n_kp in list:
                vector = votefunc((225, 110), keyPoints[n_kp.imgIdx][n_kp.trainIdx], kp)
                vector = (np.int(vector[0] / 5), np.int(vector[1] / 5))

                if (vector[0] >= 0) & (vector[1] >= 0) & (vector[0] < (imgTest.shape[0] / 5 -1)) & (vector[1] < (imgTest.shape[1] / 5 -1)):
                    votaciones[vector[0]][vector[1]] += 1

    coords = np.unravel_index(votaciones.argmax(), votaciones.shape)

    for i in range(imgTest.shape[0]):
        for j in range(imgTest.shape[1]):
            imgTest[coords[0]*5][j] = 0
            imgTest[coords[0]*5 +5][j] = 0
            imgTest[i][coords[0]*5] = 0
            imgTest[i][coords[0]*5 +5] = 0

    cv2.imshow(str(x), imgTest)
    cv2.waitKey()
