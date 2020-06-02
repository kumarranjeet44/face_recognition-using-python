import cv2
import os
import numpy as np
from PIL import Image


def getImageAndLabel(path):
    print("Inside getimage function")
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imagepaths=[os.path.join(path, f) for f in os.listdir(path)]
    faccesample=[]
    Ids=[]
    for imagepath in imagepaths:
        #loading image and convert it1into gray image
        pilImage=Image.open(imagepath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        #getting id from image
        Id=int(os.path.split(imagepath)[-1].split(".")[1])
        name=os.path.split(imagepath)[-1].split(".")[0]
        #extract the the face from training image sample
        faces=detector.detectMultiScale(imageNp)
        #if image is there than append that in the list and append id also 
        for (x,y,w,h) in faces:
            faccesample.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    print(Ids)              
getImageAndLabel("dataset")


if Id in Ids:
            print("Good we are inside")
            for imagepath in imagepaths:
                ID=int(os.path.split(imagepath)[-1].split(".")[1])
                name=os.path.split(imagepath)[-1].split(".")[0]
                if Id==ID:
                    Id=name
                else:
                    Id='unknown'


#In os.path.split(name)                                                                          
#Out[668]: ('Face-Recognition/dataSet', 'face-1.1.jpg')
#In [669]: os.path.split(name)[-1]                                                                      
#Out[669]: 'face-1.1.jpg'
#In [670]: os.path.split(name)[-1].split('.')                                                           
#Out[670]: ['face-1', '1', 'jpg']
#In [671]: os.path.split(name)[-1].split('.')[1]                                                        
##Out[671]: '1'
#In [672]: int(os.path.split(name)[-1].split('.')[1])                                                   
#Out[672]: 1