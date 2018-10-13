from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import pyautogui
from scipy.spatial import distance
import os


threshold = 0.2

detect  = dlib.get_frontal_face_detector()
path=os.getcwd()
path=path+ '/models/shape_predictor_68_face_landmarks.dat' 
predict = dlib.shape_predictor(path)

(left_start, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

factor=20

x,y,r=5,5,1


countl,countr=0,0
xal,xbl,rl,xar,xbr,rr=0,0,0,0,0,0
mxl,myl,mrl,mxr,myr,mrr=0,0,0,0,0,0
X,Y=0,0



p,q=0,0


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




def ear(eyex):
	A = distance.euclidean(eyex[1],eyex[5])
	B = distance.euclidean(eyex[2], eyex[4])
	C = distance.euclidean(eyex[0], eyex[3])

	eard = (A+B)/(2.0 * C)

	return eard




def eye(cimg,xa,xb,r,count,mx,my,mr,eye_hull):
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,4,
                            param1=230,param2=1,minRadius=0,maxRadius=45)
    
    #global xa,xb,r
    try:
                

                circles = np.uint16(np.around(circles))
                a,b,c=smallest(circles,cimg)
                a=eye_hull[4][0][0]+a
                b=eye_hull[4][0][1]+b
                if(c>0):
                    xa=xa+a
                    xb=xb+b
                    r=c+r
                    count=count+1
                    
                if(count==5):
                    mx=int(xa/5)
                    my=int(xb/5)
                    mr=int(r/5)
                    
                    count=0
                    xa=0
                    xb=0
                    r=0
                if(mx==0):
                    mx,my,mr=xa,xb,r
                
                
                cv2.circle(frame,(a,b),c,(0,255,0),2)
    except Exception as e:
        print(e)
    return xa,xb,r,count,mx,my,mr

Mcount=0
cordsx=[]
cordsy=[]
k=0
Mx,My,Myy,Mxx=0,0,0,0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    #global countl,countr
    #global xal,xbl,rl,xar,xbr,rr
    #global mxl,myl,mrl,mxr,myr,mrr
    Mcount+=Mcount
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        cv2.drawContours(frame,[shape],-1,(255,0,255),1)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame,[left_eye_hull], -1, (255,0,255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (255,0,255), 1)

        xL=int((left_eye_hull[0][0][0]+left_eye_hull[3][0][0])/2)
        yL=int((left_eye_hull[0][0][1]+left_eye_hull[3][0][1])/2)

        xR=int((right_eye_hull[0][0][0]+right_eye_hull[3][0][0])/2)
        yR=int((right_eye_hull[0][0][1]+right_eye_hull[3][0][1])/2)

        eye_width=int((left_eye_hull[0][0][0]-left_eye_hull[3][0][0]))
        eye_height=eye_width*0.28

        RxFact=1980/eye_width
        RyFact=1024/eye_height
        try:
    
            left=frame[left_eye_hull[4][0][1]:left_eye_hull[1][0][1],left_eye_hull[4][0][0]:left_eye_hull[1][0][0]]

            left =cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
            left = cv2.GaussianBlur(left,(5,5),0)

            left=cv2.equalizeHist(left)

            right=frame[right_eye_hull[4][0][1]:right_eye_hull[1][0][1],right_eye_hull[4][0][0]:right_eye_hull[1][0][0]]
        
            right =cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
            right = cv2.GaussianBlur(right,(5,5),0)

            right=cv2.equalizeHist(right)
        except Exception as e:
            print(e)
        
        
        try:

                leftEar = ear(left_eye)
                rightEar = ear(right_eye)
                ratio = (leftEar + rightEar) / 2.0
                if(Mcount%4==0):
                    blink_count=0
                    
                    
	        # This will come in the for loop
                if(ratio < threshold):
                    blink_count = blink_count + 1
                    #print("BLINKING BEEEEEP")
                if(blink_count>=1): 
                    pyautogui.click()

		
                xar,xbr,rr,countr,mxr,myr,mrr=eye(right,xar,xbr,rr,countr,mxr,myr,mrr,right_eye_hull)
                xal,xbl,rl,countl,mxl,myl,mrl=eye(left,xal,xbl,rl,countl,mxl,myl,mrl,left_eye_hull)
                    
                Rightx=abs((xR-mxr))
                Righty=abs((yR-myr))
                Leftx=abs((xL-mxl))
                Lefty=abs((yL-myl))

                Ravg=(Rightx+Leftx)/2
                Lavg=(Righty+Lefty)/2

                Mxx=int(1980/2+RxFact*Ravg)
                Myy=int(1024/2+RyFact*Lavg)
                if(Mxx>1980):
                    Mxx=1980
                if(Myy>1024):
                    Myy=1024
                

                if(p==0):
                    p,q=mxl,myl
                Mxx=(mxl-p)*10
                Myy=(myl-q)*10

                p=mxl
                q=myl
                
    
        except Exception as e:
                print(e)
        
        #cv2.circle(frame,(x,y),5,(255,0,0),3)
   #print(X,Y)
    #print(offsetx,offsety)
    
    #pyautogui.moveTo(Mxx,Myy,0)
    
    try:   
        pyautogui.moveRel(-Mxx,Myy,0.5)
    except Exception as e:
        print(e)
    cv2.imshow('Aankh',frame)
    #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
cap.release()
cap.destroyAllWindows()
cap.stop()
