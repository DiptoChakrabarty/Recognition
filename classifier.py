import os
import numpy as np 
import cv2
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
id=input("Give User ID : ")
path='dataset/user{}'.format(id)

def getImageWithID(path):
    count=0
    imagepath= [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for img_path in imagepath:
        count=count+1
        faceimg=Image.open(img_path).convert('L')
        facenp=np.array(faceimg,'uint8')
        faces.append(facenp)
        ids.append(int(id))
    return  faces,ids

faces,ids=getImageWithID(path)
recognizer.train(faces,np.array(ids))
recognizer.save('reco/trainingdata.yml')


