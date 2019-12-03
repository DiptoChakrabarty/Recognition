import cv2
import numpy as np 
cap = cv2.VideoCapture(0)
#intensity of green
blue_lower=np.array([100,150,0])
blue_upper=np.array([140,255,255])

while True:
    ret,photo= cap.read()
    img=cv2.resize(photo,(340,220))
    # convert image to HSv
    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHsv,lower,upper)
    cv2.imshow("HSV",mask)
    cv2.imshow("Normal",img)
    if (cv2.waitKey(1)==13):
        break
cap.release()
cv2.destroyAllWindows()