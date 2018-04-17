import cv2
matrix = []
for x in range(49):
    I = cv2.imread("./training_ocr/frontal_" + str(x) + ".jpg", 0)
    matrix.append(I)
    cv2.imshow('hey',I)
    cv2.waitKey()