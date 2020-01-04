import os
import numpy as np 
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


path= 'dataset'.format(id)

def getImageWith(path):
    
    imagepath= [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagepath)
    faces=[]
   # ids=[]
    for path_name in imagepath:
        #print(path_name)
      #  id=path_name.split('/')[1][-1]
        
        images= [[os.path.join(path_name,f) for f in os.listdir(path_name)]]
        #print(images)
        for im in images:
            #print(im,len(im))
            for data in im:
                faceimg=Image.open(data).convert('L')
                npimg=np.array(faceimg,'uint8')
                faces.append(npimg)
                #ids.append(int(id))
    return faces

#getImageWithID(path)

faces=getImageWith(path)
faces=np.array(faces)
cluster = KMeans(n_clusters=3)
cluster.fit(faces)
centers= cluster.cluster_centers_
labels= cluster.labels_


#check the clusters
plt.scatter(faces[:,0], faces[:,1], c=labels, cmap='rainbow')
plt.scatter(centers[:,0] ,centers[:,1], color='black')
