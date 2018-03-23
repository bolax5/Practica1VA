import cv2

frontales = cv2.CascadeClassifier('./haar/coches.xml')
matriculas = cv2.CascadeClassifier('./haar/matriculas.xml')
for x in range (1,34):


    I = cv2.imread("./testing/test" + str(x) + ".jpg", 0)
    IC = cv2.imread("./testing/test" + str(x) + ".jpg")
    faceRectFrontales = frontales.detectMultiScale(I, scaleFactor=1.01, minNeighbors=15, minSize=(100,100))
    faceRectMatriculas = matriculas.detectMultiScale(I, scaleFactor=1.01, minNeighbors=10)


    for frontal in faceRectFrontales:
        IC = cv2.rectangle(IC, (frontal[0],frontal[1]+frontal[3]), (frontal[0]+frontal[2],frontal[1]), (255,0,0), 3)
    for matricula in faceRectMatriculas:
        IC = cv2.rectangle(IC, (matricula[0],matricula[1]+matricula[3]), (matricula[0]+matricula[2],matricula[1]), (0,255,0), 3)
    cv2.imshow("",IC)
    cv2.waitKey()
