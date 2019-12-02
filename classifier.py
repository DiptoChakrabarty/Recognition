import cv2
import numpy as np 
import os

cap=cv2.VideoCapture(0)
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id= input("Enter USer id : ")
os.system("mkdir -p dataset/user{}".format(id))
count=0
while True:
    ret,photo= cap.read()
    gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    faces=  face.detectMultiScale(gray)
   
    for (x,y,w,h) in faces:
        count=count+1
        img_path="./dataset/user{}/{}.png".format(id,count)
        print(img_path)
        cv2.imwrite(img_path,gray[y:y+h,x:x+w])
        cv2.rectangle(photo,(x,y),(x+w,y+h),(0,0,255))
    cv2.imshow("click",photo)
    if(count==15):
        count=0
        break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()