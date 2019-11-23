import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np


#Ouverture du flux video
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read() # Passe à l'image suivante
ret, frame2 = cap.read()

x,y,z=frame1.shape

nextimYuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV) 
index=1;
imYuv = cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV) # Passage en niveaux de yuv

while(ret):
	hist= cv2.calcHist([imYuv], [1, 2], None, [256]*2, [-100,255, 0,255])
	hist=cv2.normalize(hist, hist)
	hist=log(hist)
	fig, ax =plt.subplots()
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
	ret, frame2 = cap.read()
	if (ret):
		nextimYuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV)    

   
cap.release()
cv2.destroyAllWindows()
