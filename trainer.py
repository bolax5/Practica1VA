import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class ocr:
    def __init__(self):
        clases = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','ESP','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        e = clases *250
        e = sorted(e)
        found = []
        images = glob.glob("./training_ocr/*.jpg")
        for k in images:
            crop_img = cv2.imread(k, 0)
            blur = cv2.bilateralFilter(crop_img, 1, 750, 7, borderType=cv2.BORDER_CONSTANT)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            im2, contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            just1valid = 0

            for cnt in contours:
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
                if ((cv2.contourArea(cnt)>=4) &((w1/h1) < 1) & (not just1valid) ) :

                    #print ("=> ", cv2.contourArea(cnt),  w1/h1 )
                    crop_img2 = th3[y1:y1 + h1, x1:x1 + w1]
                    ret4, th4 = cv2.threshold(crop_img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    cols, rows = th4.shape

                    resized = cv2.resize(th4, None, fx=10 / cols, fy=10 / cols,
                                         interpolation=cv2.INTER_LINEAR)
                    left = 10 - resized.shape[1]
                    resized = np.pad(resized, [(0, 0), (left, 0)], mode='constant', constant_values=255)
                    if (just1valid):
                        print (k, "-->", len(contours))
                        #plt.imshow(resized,cmap='gray')
                        #plt.show()
                    just1valid = 1
                    flattened = resized.flatten()


                    #print (len(flattened))

                    #plt.imshow(flattened,cmap='gray')
                    #plt.show()
                    found.append(flattened)

                #else:
                    #cv2.rectangle(th3, (x1, y1), (x1 + w1, y1 + h1), 150, 2)
                    #print (cv2.contourArea(cnt),"??", w1/h1)
            if(not just1valid):
                #x1, y1, w1, h1 = cv2.boundingRect(contours[0])
                #crop_img5 = th3[y1:y1 + h1, x1:x1 + w1]
                #plt.imshow(crop_img5,cmap='gray')
                #plt.show()
                print (k, "<<<>>>", len(contours), "------", w1/h1, cv2.contourArea(contours[0]) )

            #print ("=======================")
        clf = LinearDiscriminantAnalysis()
        clf.fit(found, e)
        self.predict = clf.predict
        self.testFound = found[::250]
        print()
    def predict(self, toPredict):
        return self.predict(toPredict)

