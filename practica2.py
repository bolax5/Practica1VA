import cv2
import numpy as np
from functools import reduce
import glob
from scipy import ndimage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from string import maketrans
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
                    crop_img2 = th3[y1:y1 + h1, x1:x1 + w1]
                    ret4, th4 = cv2.threshold(crop_img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    cols, rows = th4.shape
                    resized = cv2.resize(th4, None, fx=10 / cols, fy=10 / cols, interpolation=cv2.INTER_LINEAR)
                    left = 10 - resized.shape[1]
                    resized = np.pad(resized, [(0, 0), (left, 0)], mode='constant', constant_values=255)
                    just1valid = 1
                    flattened = resized.flatten()
                    found.append(flattened)
        clf = LinearDiscriminantAnalysis()
        clf.fit(found, e)
        clf.predict(found)
        self.predict = clf.predict
        self.testFound = found[::50]



def filter_by_typical_desviation(contours, max_differ):
    medias = list(map(lambda x: cv2.contourArea(x), contours))
    media = reduce((lambda x, y: x + y), medias)
    media = media / len(contours)
    differ = ((media + (max_differ / 100) * media), (media - (max_differ / 100) * media))
    return list(filter(lambda x: (cv2.contourArea(x) <= differ[0]) & (cv2.contourArea(x) >= differ[1]), contours))

def inspect_plate(crop_img):
    im2, contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    not_so_diferent = filter_by_typical_desviation(contours, 90)
    found = []
    not_so_different_sorted = []
    for cnt in not_so_diferent:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if  ((((w1 / h1) <= 0.8) & ((w1 / h1) >= 0.55)) | (((w1 / h1) <= 0.4) & ((w1 / h1) >= 0.2))):
            not_so_different_sorted.append((x1,y1,w1,h1))
    not_so_different_sorted = sorted(not_so_different_sorted, key=lambda tup: tup[0])
    lastX = 0
    for x1, y1, w1, h1 in not_so_different_sorted:
        if lastX <= (x1):
            crop_img2 = crop_img[y1:y1 + h1, x1:x1 + w1]
            #cv2.imshow("", crop_img2)
            #cv2.waitKey()
            cols, rows = crop_img2.shape
            ret5, th5 = cv2.threshold(crop_img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            resized = cv2.resize(th5, None, fx=10 / cols, fy=10 / cols, interpolation=cv2.INTER_LINEAR)
            left = 10 - resized.shape[1]
            resized = np.pad(resized, [(0, 0), (left, 0)], mode='constant', constant_values=255)
            #plt.imshow(resized)
            #plt.show()
            flattened = resized.flatten()
            found.append((x1, flattened))
            lastX = x1+ w1

    if (len(found) > 1):
        toPredict = tuple([list(tup) for tup in zip(*found)])[1]
        return trainer.predict(toPredict)
    else:
        return []


def check_umbral(crop_img):
    equ = cv2.equalizeHist(crop_img)
    blur = cv2.bilateralFilter(equ, 1, 750, 7, borderType=cv2.BORDER_CONSTANT)

    foo, th0_255 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    foo, th100_165 = cv2.threshold(blur, 100, 165, cv2.THRESH_BINARY_INV)
    foo, th_separador = cv2.threshold(th100_165, 0, 255, cv2.THRESH_OTSU)

    foo, th50_110 = cv2.threshold(blur, 50, 110, cv2.THRESH_BINARY_INV)
    foo, th_extremo = cv2.threshold(th50_110, 0, 255, cv2.THRESH_OTSU)

    return th0_255, th_separador, th_extremo
def process(plate):

    concat = reduce((lambda x, y: x+y), plate)
    intab = "OI"
    outtab = "01"
    trantab = maketrans(intab, outtab)
    processed = concat.translate(trantab)

    return processed



trainer = ocr()
images = glob.glob("./testing_ocr/*.jpg")
matriculas = cv2.CascadeClassifier('./haar/matriculas.xml')
output_file = open("output.txt","w")

for k in images:
    I = cv2.imread(k, 0)
    faceRectMatriculas = matriculas.detectMultiScale(I, scaleFactor=1.07, minNeighbors=6)
    for (x, y, w, h) in faceRectMatriculas:
        crop_img = I[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        a,b,c = check_umbral(crop_img)

        plate_numbers_a = inspect_plate(a)

        plate_numbers_b = inspect_plate(b)

        plate_numbers_c = inspect_plate(c)

        if len(plate_numbers_a) > len(plate_numbers_b):
            if len(plate_numbers_a) > len(plate_numbers_c):
                final = plate_numbers_a
            else:
                final = plate_numbers_c
        elif len(plate_numbers_b) > len(plate_numbers_c):
                final = plate_numbers_b
        else:
            final = plate_numbers_c
        final_procesado = process(final)

        output_file.write('Imagen: {0}, Centro: {1}, Matricula: {2}, Largo/2: {3}\n'.format(k, (x+(w/2),y+(h/2)),final_procesado,w/2))




