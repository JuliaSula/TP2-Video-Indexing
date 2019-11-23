import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from scipy import signal


#Ouverture du flux video
cap = cv2.VideoCapture('../TP2_Videos/1.png')
ret, frame1 = cap.read() # Passe a l'image suivante
ret, frame2 = cap.read()
ret=1
frame1=cv2.imread('../TP2_Videos/2.jpg')
frame2=cv2.imread('../TP2_Videos/2.jpg')
x,y,z=frame1.shape

nextimYuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV) 
index=1;
imYuv = cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV) # Passage en niveaux de yuv
y=[]

while(ret):
    #Calcul histogrammes YUV
	hist= cv2.calcHist([imYuv], [1, 2], None, [32]*2, [0,255, 0,255])
	hist=cv2.normalize(hist, hist)
	hist1= cv2.calcHist([nextimYuv], [1, 2], None, [32]*2, [0,255, 0,255])
	hist1=cv2.normalize(hist1, hist1)
	
    #Correlation entre histogrammes
	y.append(cv2.compareHist(hist, hist1,cv2.HISTCMP_CORREL))

	#Plot de la correlation
	fig2, ax2 =plt.subplots()	
	ax2 = plt.gca()
	ax2.plot(y)
	plt.xlabel("Index")
	plt.ylabel("Corrélation entre frames")
	plt.draw()
	fig2.canvas.draw()
	imgt = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
	imgt  = imgt.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
	imgt = cv2.cvtColor(imgt,cv2.COLOR_RGB2BGR)
	cv2.imshow("Corrélation entre frames",imgt)

    #Plot de l`histogrammes
	fig, ax =plt.subplots()
	ax.set_xlim([0, 32 - 1])
	ax.set_ylim([0, 32 - 1])
	im = ax.imshow(hist)
	fig.colorbar(im)
	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
	img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

	# display image with opencv or any operation you like
	cv2.imshow("plot",img)
	cv2.imshow('Image et Champ de vitesses (Farnebäck)',frame2)
	k = cv2.waitKey(15) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('Frame_%04d.png'%index,frame2)
		cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
	imYuv = nextimYuv
	frame2=cv2.imread('../TP2_Videos/2.jpg')
	#ret, frame2 = cap.read()
	if (ret):
		nextimYuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV)    
	
   
cap.release()
cv2.destroyAllWindows()
