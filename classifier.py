import cv2
import numpy as np 

cap=cv2.VideoCapture(0)
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,photo= cap.read()
    faces=  face.detectMultiScale(photo)
    for (x,y,w,h) in faces:
        cv2.rectangle(photo,(x,y),(x+w,y+h),(0,0,255))
    cv2.imshow("click",photo)
    if(cv2.waitKey(1)==13):
        break
cap.release()
cv2.destroyAllWindows()