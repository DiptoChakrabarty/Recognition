import os
import numpy as np 
import cv2
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


path= './dataset/all'

def getImage(path_name):
    
    faces=[]
        
    images_path= [os.path.join(path_name,f) for f in os.listdir(path_name)]
    print(images_path)
    for im in images_path:
        #print(im,len(im))
        image=cv2.imread(im)

        

        image=cv2.resize(image,(224,224))

        faces.append(image)
           
            
            
                
    faces=np.array(faces)
    return faces

#getImageWithID(path)

faces=getImage(path)

print(type(faces))
print(faces)
print(faces.shape)
print(faces[0].shape)

faces=faces.reshape(len(faces),-1)
faces=faces.astype(float)/255.0
print(faces.shape)
print(faces[0].shape)

#faces=np.array(faces,dtype=object)


#faces=pd.DataFrame(faces)
cluster = KMeans(n_clusters=3)
cluster.fit(faces)
centers= cluster.cluster_centers_
labels= cluster.labels_


#check the clusters

plt.scatter(faces[:,0], faces[:,1], c=labels, cmap='rainbow')
plt.scatter(centers[:,0] ,centers[:,1], color='black')
