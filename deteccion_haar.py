import cv2

cascade = cv2.CascadeClassifier('./haar/coches.xml')
I = cv2.imread("./testing/test1.jpg", 0)
faceRect = cascade.detectMultiScale(I, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
print (faceRect)