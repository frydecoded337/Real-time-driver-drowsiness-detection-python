import cv2
import dlib
from imutils import face_utils
import math
import matplotlib.pyplot as plt
import playsound
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def eye_aspect_ratio(points):
    [a,b,c,d,e,f]=points
    v1=distance(b,f)
    v2=distance(c,e)
    h=distance(a,d)
    return (v1+v2)/(2*h)
    
def distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
def alarm_funct():
    if alarm==True:
        playsound.playsound('alarm.mp3')  

cap = cv2.VideoCapture(0)
blink=0
EAR=[]
threshold=0.25
count=0
num_frame=5
while True:
    valid, image = cap.read()                       # Getting image(frame) from webcam 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting the image to gray scale
    rects = detector(gray, 1)                       # detect face in the gray image
   
    left=[i for i in range (36,42)]                 
    right=[i for i in range (42,48)]
    left_eye=[]
    right_eye=[]
      
    for rect in rects:                              # finding the landmarks on each face 
        shape = predictor(gray, rect)              
        shape = face_utils.shape_to_np(shape)  
        
        left_eye=[shape[i] for i in left]           # coordinates corresponding to left and right eye
        right_eye=[shape[i] for i in right]         
         
        for eye in [left_eye, right_eye]:
            for x,y in eye:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)# draw circles on the eye landmarks detected    
           
        left_EAR=eye_aspect_ratio(left_eye)         #eye_aspect_ratio
        right_EAR=eye_aspect_ratio(right_eye)
        
        avg_EAR=(left_EAR+right_EAR)/2.0            # average eye aspect ratio
        EAR.append(avg_EAR)
    
    plt.plot(EAR, 'r-') 
    
    if avg_EAR<threshold:
        count+=1
        if count>num_frame:
            alarm=True
            #print(alarm)
            count=0
    
    else:
        blink+=1
        if blink==3:
            count=0
        alarm=False
   
    cv2.imshow("Output", image)                     # display the image with landmarks  
    key = cv2.waitKey(5) 
    if key == 27:                                   # check whether ESC is pressed 
        break

cv2.destroyAllWindows()
cap.release()
