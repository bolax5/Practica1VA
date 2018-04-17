import cv2
import numpy as np
matriculas = cv2.CascadeClassifier('./haar/matriculas.xml')

for x in range (1,48):
    I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
    faceRectMatriculas = matriculas.detectMultiScale(I, scaleFactor=1.01, minNeighbors=10)
    for (x, y, w, h) in faceRectMatriculas:
        crop_img = I[y:y+h, x:x+w]
        blur = cv2.bilateralFilter(crop_img,1,750,7,borderType=cv2.BORDER_CONSTANT)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("", th3)
        cv2.waitKey()


        im2, contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            if (((w1/h1)<=0.8) & ((w1/h1)>=0.55 )) | (((w1/h1)<=0.4) & ((w1/h1)>=0.3 )) :
                cv2.rectangle(crop_img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                #cv2.drawContours(crop_img, cnt, -1, 0, 1)
                print ("=> ", cv2.contourArea(cnt),  w1/h1 )
                crop_img2 = crop_img[y1:y1 + h1, x1:x1 + w1]
                cv2.imshow("", crop_img2)
                cv2.waitKey()
            elif((w1/h1) <1 ):
                print (cv2.contourArea(cnt),"??", w1/h1)


        cv2.imshow("", I )
        cv2.waitKey()
        print ('####################')
