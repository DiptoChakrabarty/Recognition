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
x,y,z=photo.shape
photo2=photo.reshape(x*y,z)


#Use Kmeans
cluster = KMeans(n_clusters=3)
cluster.fit(photo2)
centers= cluster.cluster_centers_
labels= cluster.labels_

#plot image

plt.figure(figsize=(15,8))
plt.imshow((centers[labels].reshape(x,y,z)*255).astype(np.uint8))
