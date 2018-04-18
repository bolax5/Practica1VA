import cv2
import numpy as np
from operator import itemgetter
import glob
images = glob.glob("./testing_ocr/*.jpg")

matriculas = cv2.CascadeClassifier('./haar/matriculas.xml')

for k in images:
    I = cv2.imread(k, 0)
    faceRectMatriculas = matriculas.detectMultiScale(I, scaleFactor=1.01, minNeighbors=10)
    for (x, y, w, h) in faceRectMatriculas:
        crop_img = I[y:y+h, x:x+w]
        blur = cv2.bilateralFilter(crop_img,1,750,7,borderType=cv2.BORDER_CONSTANT)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("", th3)
        cv2.waitKey()


        im2, contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #sorted(contours, key=itemgetter(0));
        found = []
        for cnt in contours:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            if (((w1/h1)<=0.8) & ((w1/h1)>=0.55 )) | (((w1/h1)<=0.4) & ((w1/h1)>=0.3 )) :
                cv2.rectangle(th3, (x1, y1), (x1 + w1, y1 + h1), 0, 1)
                #cv2.drawContours(crop_img, cnt, -1, 0, 1)
                #print ("=> ", cv2.contourArea(cnt),  w1/h1 )
                crop_img2 = th3[y1:y1 + h1, x1:x1 + w1]
                found.append((x1,crop_img2))

            else:
                cv2.rectangle(th3, (x1, y1), (x1 + w1, y1 + h1), 150, 2)
                #print (cv2.contourArea(cnt),"??", w1/h1)
        found =  sorted(found, key=lambda tup:tup[0]);
        #for tupla in found:
            #cv2.imshow("", tupla[1])
            #cv2.waitKey()
        cv2.imshow("", th3 )
        cv2.waitKey()
        print ('####################')
