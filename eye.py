import numpy as np
import cv2




cap=cv2.VideoCapture(0)

eyeCascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
    ret, img =cap.read()

    #img=cv2.imread('xd.jpg')
    gimg =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gimg=cv2.equalizeHist(gimg)
    faces = faceCascade.detectMultiScale(gimg, 1.1, 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(gimg,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gimg[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    cv2.imshow('xxd',gimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
