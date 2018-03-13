import cv2
m = []
for x in range (1,5):
    I = cv2.imread("./training/frontal_" + str(x) + ".jpg", 0)
    m.append(I)
orb = cv2.ORB_create(nfeatures=100, nlevels=4)
for imagen in m:
    pts = orb.detect(imagen, None)
    pts, des = orb.compute(imagen, pts)
    img2 = cv2.drawKeypoints(imagen, pts, None, color=(0, 255, 0), flags=0)
    cv2.imshow("ventana",img2)
    cv2.waitKey()
