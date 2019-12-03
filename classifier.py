import os
import numpy as np 
import cv2
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path='dataset'.format(id)

def getImageWithID(path):
    count=0
    imagepath= [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagepath)
    faces=[]
    ids=[]
    for path_name in imagepath:
        #print(path_name)
        id=path_name.split('/')[1][-1]
        
        images= [[os.path.join(path_name,f) for f in os.listdir(path_name)]]
        #print(images)
        for im in images:
            #print(im,len(im))
            for data in im:
                faceimg=Image.open(data).convert('L')
                npimg=np.array(faceimg,'uint8')
                faces.append(npimg)
                ids.append(int(id))
    return faces,ids

#getImageWithID(path)

faces,ids=getImageWithID(path)
recognizer.train(faces,np.array(ids))
recognizer.save('reco/model.yml')
print("Completed Operation")