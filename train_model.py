import os
import cv2
import tkinter as tk
from tkinter import *
import time
import numpy as np
from PIL import Image
window=tk.Tk()
window.title("Appointment Through FaceRecognition:-A model by CSE Branch students 2K16(Group 04)")
window.geometry('1250x720')
window.configure()
#input images to datasets
def take_image():
	cam=cv2.VideoCapture(0)
	detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	ID=txt.get()
	Name=txt2.get()
	sample=0
	while(True):
		ret,img=cam.read()
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces=detector.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			sample=sample+1
			#now we will save the frame into dataset folder
			cv2.imwrite("dataset/"+Name+"."+ID+"."+str(sample)+".jpg",gray[y:y+h,x:x+w])
			cv2.imshow("frame",img)
			#wait for 100 ms
		if cv2.waitKey(1) & 0xff==ord('q'):
			break#break if sample no is more than 100
		elif sample>100:
			break
	cam.release()
	cv2.destroyAllWindows()
	res="Image Saved Successfully Id is:-"+ ID +" Name is:-" +Name
	Notification.configure(text=res,fg="springGreen3",width=50,font=('time',18,'bold'))
	Notification.place(x=250,y=400)

path='dataset'
Id_name=[]
imagepaths=[os.path.join(path, f) for f in os.listdir(path)]
def idname():
	for imagepath in imagepaths:
		ID=int(os.path.split(imagepath)[-1].split(".")[1])
		name=os.path.split(imagepath)[-1].split(".")[0]
		Id_name.append(name) 
	return Id_name

#For model training 
def training():
		recognizer=cv2.face.LBPHFaceRecognizer_create()
		faces, Id=getImageAndLabel("dataset")
		recognizer.train(faces, np.array(Id))
		recognizer.save('model/trained_model2.yml')
		
		res="Model Trained"
		Notification.configure(text=res,fg="springGreen3",width=50,font=('time',18,'bold'))
		Notification.place(x=250,y=400)
def getImageAndLabel(path):
	imagepaths=[os.path.join(path, f) for f in os.listdir(path)]
	detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	faccesample=[]
	Ids=[]
	for imagepath in imagepaths:
		#loading image and convert it into gray image
		pilImage=Image.open(imagepath).convert('L')
		imageNp=np.array(pilImage,'uint8')
		#getting id from image
		Id=int(os.path.split(imagepath)[-1].split(".")[1])
		#extract the the face from training image sample
		faces=detector.detectMultiScale(imageNp)
		#if image is there than append that in the list and append id also 
		for (x,y,w,h) in faces:
			faccesample.append(imageNp[y:y+h,x:x+w])
			Ids.append(Id)				
	return faccesample,Ids
window.grid_rowconfigure(0,weight=1)
window.grid_columnconfigure(0,weight=1)
def on_closing():
	from tkinter import messagebox

	if messagebox.askokcancel("Quit","Do You Want to Quit"):
		window.destroy()
window.protocol("WM_DELETE_WINDOW",on_closing)
Notification=tk.Label(window,text="All things good",bg='Green',fg="white",width=15,height=3)
lbl1=tk.Label(window,text="Enter Id",fg='black',width=20,height=3 ,font=('time',30,'italic bold'))
lbl1.place(x=169,y=150) 

def testVal(inStr,acttype):
	if acttype=='1':
		if not inStr.isdigit():
			return False
	return True	
message=tk.Label(window,text='Appointment through FaceRecognition Model',bg="cyan",fg="black",width=50,height=3,font=('time',30,'italic bold'))		
message.place(x=80,y=20)
txt=tk.Entry(window,validate="key",width=20,fg="red")
txt['validatecommand']=(txt.register(testVal),'%P','%d')
txt.place(x=575,y=210)
lbl2=tk.Label(window,text="Enter Name",fg="black",width=20,height=3,font=('time',30,'italic bold'))
lbl2.place(x=200,y=250)
txt2=tk.Entry(window,width=20,fg="red")
txt2.place(x=580,y=310)

takeImg=tk.Button(window,text="Take Image",command=take_image,fg='black',bg='green',width=15,height=2,activebackground='Red',font=('time',20,'italic bold'))
takeImg.place(x=200,y=500)
trainImg=tk.Button(window,text="Train Image",command=training,fg='black',bg='green',width=15,height=2,activebackground='Red',font=('time',20,'italic bold'))
trainImg.place(x=700,y=500)

window.mainloop()