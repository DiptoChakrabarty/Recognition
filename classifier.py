import cv2
import numpy as np 

cap=cv2.VideoCapture(0)
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id= input("Enter USer id : ")
count=0
while True:
    ret,photo= cap.read()
    gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    faces=  face.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite("dataset/user" + str(id)+ str(count) + ".png",gray[x:x+w,y:y+h])
        cv2.rectangle(photo,(x,y),(x+w,y+h),(0,0,255))
        cv2.waitKey(30)
    cv2.imshow("click",photo)
    if(count==20):
        count=0
        break
cap.release()
cv2.destroyAllWindows()