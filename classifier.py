import os
import numpy as np 
import cv2
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
id=input("Give User ID : ")
path='dataset/user{}'.format(id)

def getImageWithID(path):
    imagepath= [os.path.join(path,f) for f in os.listdir(path)]
    print(imagepath)


getImageWithID(path)