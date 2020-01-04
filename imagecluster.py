import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
cap = cv2.VideoCapture(0)
ret,photo = cap.read()
cv2.imshow("photo",photo)
cv2.waitKey()
cv2.destroyAllWindows()
cap.release()
# Reshape Image
#photo=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
rgb=cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)

x,y,z=rgb.shape
rgb=rgb.reshape(x*y,z)


#Use Kmeans
rgb=np.float32(rgb)
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)



#plot image

plt.figure(figsize=(15,8))
plt.imshow((centers[labels].reshape(x,y,z)*255).astype(np.uint8))
