import cv2

frontales = cv2.CascadeClassifier('./haar/coches.xml')
matriculas = cv2.CascadeClassifier('./haar/matriculas.xml')
video1 = cv2.VideoCapture('./Videos/video2.wmv')
while (video1.isOpened()):

    foo, IC = video1.read()
    I = cv2.cvtColor(IC, cv2.COLOR_BGR2GRAY)
    faceRectFrontales = frontales.detectMultiScale(I, scaleFactor=1.02, minNeighbors=5, minSize=(50,50))
    faceRectMatriculas = matriculas.detectMultiScale(I, scaleFactor=1.2, minNeighbors=2)


    for frontal in faceRectFrontales:
        IC = cv2.rectangle(IC, (frontal[0],frontal[1]+frontal[3]), (frontal[0]+frontal[2],frontal[1]), (255,0,0), 2)
    for matricula in faceRectMatriculas:
        IC = cv2.rectangle(IC, (matricula[0],matricula[1]+matricula[3]), (matricula[0]+matricula[2],matricula[1]), (0,255,0), 2)
    cv2.imshow("",IC)
    cv2.waitKey()