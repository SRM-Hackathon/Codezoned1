from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
#import pyautogui


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(left_start, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)



x,y,r=5,5,1
def smallest(circles,cimg):
    least=10000000

    global x,y,r
    for i in circles[0,:]:
        #print("CIRCLE")
        s=0
        
        a=np.array((i[0],i[1]))
        for k in range(0,cimg.shape[0]):
            for l in range(0,cimg.shape[1]):
                b=np.array((k,l))
                
                dist=np.linalg.norm(a-b)
                #print("Distance = ",dist,"\n Radius =",i[2])
                if(dist <= i[2]):
                    s=s+cimg[k][l]
        #print(s)
        if(s<least):
            least=s
            global x
            x=i[0]
            global y
            y=i[1]
            global r
            r=i[2]

        #cv2.circle(cimg,(x,y),r,(255,0,0),2)
        return x,y,r
        #cv2.circle(cimg,(x,y),250,(255,0,0),2)
count=0
xa,xb,r=0,0,0
mx,my,mr=0,0,0
cx,cy=0,0
no=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape2 = face_utils.shape_to_np(shape)
        left_eye = shape2[left_start:left_end]
        right_eye = shape2[right_start:right_end]
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame,[left_eye_hull], -1, (255,0,255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (255,0,255), 1)

        x=int((left_eye_hull[0][0][0]+left_eye_hull[3][0][0])/2)
        y=int((left_eye_hull[0][0][1]+left_eye_hull[3][0][1])/2)
        cimg=frame[left_eye_hull[4][0][1]:left_eye_hull[1][0][1],left_eye_hull[4][0][0]:left_eye_hull[1][0][0]]
        cimg =cv2.cvtColor(cimg,cv2.COLOR_RGB2GRAY)
        cimg=cv2.equalizeHist(cimg)
        
        
        try:
                circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,4,
                            param1=100,param2=1,minRadius=0,maxRadius=45)

                circles = np.uint16(np.around(circles))
                a,b,c=smallest(circles,cimg)
                a=left_eye_hull[4][0][0]+a
                b=left_eye_hull[4][0][1]+b
                if(c>=2):
                    xa=xa+a
                    xb=xb+b
                    r=c+r
                    count=count+1
                    #print(count)
                if(count==3):
                    mx=int(xa/3)
                    my=int(xb/3)
                    mr=int(r/3)
                    #print("XA=",xa,"Xb= ",xb,"Radius = ",r)
                    
                    count=0
                    xa=0
                    xb=0
                    r=0
                if(mx==0):
                    mx,my,mr=xa,xb,r
                no=no+1
                cx=cx+mx
                cy=my+cy
                print("X= ",cx,"Y =",cy)
                if(no==100):
                    print("enterxd")
                    no=input()
                    no=0
                    cx=0
                    cy=0
                cv2.circle(frame,(mx,my),mr,(255,0,0),2)
                

        except Exception as e:
                print(e)
        
        #cv2.circle(frame,(x,y),5,(255,0,0),3)
		
    cv2.imshow('Aankh',frame)
    #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cap.destroyAllWindows()
cap.stop()
