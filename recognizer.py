import cv2,os
import numpy as np
import train_model as tm
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model2.yml')
cascadepath="haarCascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascadepath)
font=cv2.FONT_HERSHEY_SIMPLEX
path='dataset'
cap=cv2.VideoCapture(0)

Id_name=tm.idname()
while True:
	ret,im=cap.read()
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	faces=faceCascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		Id,conf=recognizer.predict(gray[y:y+h,x:x+w])
		cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),7)
		if(Id==1):
			Id='Ranjeet'
		cv2.putText(im,str(Id),(x,y-40),font,2,(255,255,255),3)
		cv2.putText(im,str(conf),(x,y),font,2,(255,255,255),3)

	cv2.imshow("im",im)
	if cv2.waitKey(10) & 0xFF==ord("q"):
		break
		
cap.release()
cv2.destroyAllWindows()			