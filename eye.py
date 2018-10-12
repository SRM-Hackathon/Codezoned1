import numpy as np
import cv2


x,y,r=5,5,1
def smallest(circles,cimg):
    least=10000000

    global x,y,r
    for i in circles[0,:]:
        print("CIRCLE")
        s=0
        
        a=np.array((i[0],i[1]))
        for k in range(0,cimg.shape[0]):
            for l in range(0,cimg.shape[1]):
                b=np.array((k,l))
                
                dist=np.linalg.norm(a-b)
                #print("Distance = ",dist,"\n Radius =",i[2])
                if(dist <= i[2]):
                    s=s+cimg[k][l]
        print(s)
        if(s<least):
            least=s
            global x
            x=i[0]
            global y
            y=i[1]
            global r
            r=i[2]

        #cv2.circle(cimg,(x,y),r,(255,0,0),2)
        cv2.circle(cimg,(x,y),2,(255,0,0),3)
cap=cv2.VideoCapture(0)

eyeCascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
x=True
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
            cimg=roi_gray[ey:ey+eh,ex:ex+ew]

            circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,10,
                            param1=250,param2=12,minRadius=0,maxRadius=45)
            try:


                circles = np.uint16(np.around(circles))
                smallest(circles,cimg)

            except AttributeError:
                print("fix this")

    #x=Falsc
    cv2.imshow('xxd',gimg)
    #print("what")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("what")
        break
cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2


x,y,r=5,5,1
def smallest(circles,cimg):
    least=10000000

    global x,y,r
    for i in circles[0,:]:
        print("CIRCLE")
        s=0
        
        a=np.array((i[0],i[1]))
        for k in range(0,cimg.shape[0]):
            for l in range(0,cimg.shape[1]):
                b=np.array((k,l))
                
                dist=np.linalg.norm(a-b)
                #print("Distance = ",dist,"\n Radius =",i[2])
                if(dist <= i[2]):
                    s=s+cimg[k][l]
        print(s)
        if(s<least):
            least=s
            global x
            x=i[0]
            global y
            y=i[1]
            global r
            r=i[2]

        #cv2.circle(cimg,(x,y),r,(255,0,0),2)
        cv2.circle(cimg,(x,y),2,(255,0,0),3)
cap=cv2.VideoCapture(0)

eyeCascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
x=True
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
            cimg=roi_gray[ey:ey+eh,ex:ex+ew]

            circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,10,
                            param1=250,param2=12,minRadius=0,maxRadius=45)
            try:


                circles = np.uint16(np.around(circles))
                smallest(circles,cimg)

            except AttributeError:
                print("fix this")

    #x=Falsc
    cv2.imshow('xxd',gimg)
    #print("what")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("what")
        break
cap.release()
cv2.destroyAllWindows()
