import cv2
import numpy as np 
import os

cap=cv2.VideoCapture(0)
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("reco/model.yml")
id=0
font=font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,photo= cap.read()
    gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    faces=  face.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(photo,(x,y),(x+w,y+h),(0,0,255))
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="Dipto"
        else:
            id="Unknown"
        cv2.putText(photo,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    cv2.imshow("click",photo)
    if(cv2.waitKey(1)==13):
        
        break

cap.release()
cv2.destroyAllWindows()