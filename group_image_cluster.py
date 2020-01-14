import os
import numpy as np 
import cv2
import pandas as pd
import keras
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



faces=getImage(path)

print(type(faces))

print(faces.shape)
print(faces[0].shape)
x,y,z=faces[0].shape




# Load Vgg Models to transform into more meaningful representations

vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(x,y,z))

vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(x,y,z))

resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(x,y,z))



# Change shape of images

def order_change(faces):
    faces=faces.reshape(len(faces),-1)
    faces=faces.astype(float)/255.0

    print(faces.shape)
    print(faces[0].shape)

    return faces


# Function for models change image

def models_transform(model,images):

    pred=model.predict(images)

    flat = pred.reshape(images.shape[0]*images.shape[1],-1)

    return flat



vgg16_output = covnet_transform(vgg16_model, faces)
print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))

vgg19_output = covnet_transform(vgg19_model, faces)
print("VGG19 flattened output has {} features".format(vgg19_output.shape[1]))

resnet50_output = covnet_transform(resnet50_model, faces)
print("ResNet50 flattened output has {} features".format(resnet50_output.shape[1]))

# Code of KMeans clustering





def K_cluster(cluster,faces):

    start=time.time()
    cluster.fit(faces)
    end=time.time()
    print("Total Time Needed was",end-start)
    return cluster


print("Case of vgg16 output")

faces=order_change(vgg16_output)
cluster = KMeans(n_clusters=3)

cluster=K_cluster(cluster,faces)

centers= cluster.cluster_centers_
labels= cluster.labels_


#check the clusters
print("The Number of Centers",len(centers))
print([list(labels).count(i) for i in range(max(labels)+1)])

plt.scatter(faces[:,0], faces[:,1], c=labels, cmap='rainbow')
plt.scatter(centers[:,0] ,centers[:,1], color='black')


cv2.imshow("cluster1",centers[0].reshape(x,y,z))
cv2.imshow("cluster2",centers[1].reshape(x,y,z))
cv2.imshow("cluster3",centers[2].reshape(x,y,z))
cv2.waitKey(0)
cv2.destroyAllWindows()
