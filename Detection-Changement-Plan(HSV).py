
import matplotlib.pyplot as plt
import math
from matplotlib import ticker
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from datetime import datetime






#Ouverture du flux video
cap = cv2.VideoCapture('../TP2_Videos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v')
#cap = cv2.VideoCapture(0)
ret, frame1 = cap.read() # Passe à l'image suivante
ret, frame2 = cap.read()

width,height,z=frame1.shape

nextHSV = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV) 

index=0;
imHSV = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV) # Passage en niveaux de yuv
alpha=0.25
beta=0.7
d_inst=[]
theta1=0
theta2=6000
D=0
D_acc=0
x=[]
y=[]
plt.ion()
D_acc_g=[]
theta_a=[]
while(ret):
	factor=100
    #Resize et split en HSV
	nextHSVf=cv2.resize(nextHSV, (64,64), None, cv2.INTER_AREA)
	imHSVf=cv2.resize(imHSV, (64,64), None, cv2.INTER_AREA)
	nextH, nextS, nextV=cv2.split(nextHSVf)
	imH, imS, imV=cv2.split(imHSVf)

    #Calcul distance HSV
	dist=math.fmod((alpha)*sum(sum(nextH-imH)),2*math.pi)+(1-alpha)*abs(sum(sum(nextS-imS)))
	d_inst.append(dist)

	#Plot Distances
	D=abs(d_inst[index]-d_inst[index-1])
	plt.ion()
	y.append(D)
	plt.title("Distances")
	plt.xlabel("index")
	
	ax = plt.gca()
	ax1=plt.gca()
	ax2=plt.gca()
	ax3=plt.gca()
	
	ax.plot(y, label='Distance Reguliere')
	ax1.plot(d_inst, label='Distance instantanee')
	ax2.plot(D_acc_g, label='Distance accumulee')	
	ax3.plot(theta_a, label='Seuil')

	ax.legend()
	ax1.legend()
	ax2.legend()
	ax3.legend()

	plt.draw()
	ax.figure.canvas.draw()
	plt.pause(0.05)
	plt.cla()
	
	#Verification des seuils
	if(abs(D)>theta1):
			D_acc=0
			print("changement de plan brusque", index)
			#continue
	elif(D_acc>theta1):
			print("changement de plan progressive")
			D_acc=0
	elif(abs(dist)>theta2):
			D_acc=D_acc+abs(D)
			#print("test", D_acc, theta1)
	else :
			print("fausse alerte")
			D_acc=0	
	D_acc_g.append(D_acc)
	theta1=beta*theta1+(1-beta)*dist
	theta_a.append(theta1)
	

	cv2.imshow('Image et Champ de vitesses (Farnebäck)', nextHSVf)
	cv2.imshow('Image et Champ de vitesses (Farnebäck)', nextHSV)
	k = cv2.waitKey(15) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('Frame_%04d.png'%index,frame2)
		cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
	imHSV = nextHSV
	ret, frame2 = cap.read()
	if (ret):
		nextHSV = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)    

	

	#print(d)   
plt.show()
cap.release()
cv2.destroyAllWindows()
