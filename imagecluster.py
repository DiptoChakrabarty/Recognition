import cv2
from sklearn.cluster import Kmeans
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
ret,photo = cap.read()
cv2.imshow("photo",photo)
cv2.waitKey()
cv2.destroyAllWindows()
cap.release()
# Reshape Image
x,y,z=photo.shape
photo2=photo.reshape(x*y,z)


#Use Kmeans
cluster = Kmeans(n_cluster=7)
cluster.fit(photo2)
centers= cluster.cluster_centers_
labels= cluster.labels_

#plot image

plt.figure(figsize=(15,8))
plt.imshow(centers[labels].reshape(x,y,z))
