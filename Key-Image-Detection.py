import matplotlib.pyplot as plt
from matplotlib import ticker
import statsmodels.api
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def data_sample(y, frame1,frame2):
	print(sum(sum(frame1-frame2)))
	if(sum(sum(frame1-frame2))>10000):
		
		y.append(np.reshape(frame1,(np.shape(frame1)[0]*np.shape(frame1)[1])))
		y.append(np.reshape(frame2,(np.shape(frame1)[0]*np.shape(frame1)[1])))
	

cap = cv2.VideoCapture('../TP2_Videos/5.avi')
#Capture de frames
ret, frame1 = cap.read() 
ret, frame2 = cap.read()

#Passage de frames au niveaux de gris
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
y=[]
index=0
while(ret):
	index += 1
	cv2.imshow('Image et Champ de vitesses (Farnebäck)',frame2)
	data_sample(y, prvs, next)
	prvs = next
	ret, frame2 = cap.read()
	
	if (ret):
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
print(np.shape(y))

cap.release()
cv2.destroyAllWindows()
#dataTest = pd.read_csv(y ,  names=l, index_col=-1,  usecols=[i for i in range(65)])

kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=400, n_init=10, random_state=0)
kmeans.fit(y)
cluster=np.reshape(kmeans.cluster_centers_, (np.shape(frame1)[0],np.shape(frame1)[1]))
#cluster = cv2.cvtColor(cluster,cv2.COLOR_BGR2GRAY)
plt.imshow(cluster, cmap='brg')
plt.show()


cap = cv2.VideoCapture('../TP2_Videos/5.avi')
#Capture de frames
ret, frame1 = cap.read() 
ret, frame2 = cap.read()

#Passage de frames au niveaux de gris
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
thrs=100000
index=0
i=0
t=[]
yc=[]
while(ret):
	index += 1
	cv2.imshow('Image et Champ de vitesses (Farnebäck)',frame2)
	somme= abs(np.sum(prvs-cluster))
	print(somme)
	if(somme < thrs):
		y=prvs
		yc=frame1
		thrs=abs(np.sum(prvs-cluster))
	prvs = next
	ret, frame2 = cap.read()
	frame1=frame2
	if (ret):
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
print(np.shape(y))

cap.release()
cv2.destroyAllWindows()
img=cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
cv2.imshow('',img)
cv2.imshow('', yc)
k = cv2.waitKey(150) & 0xff
cv2.imwrite('../TP2_Videos/Image-key5.png',yc)
cv2.imwrite('../TP2_Videos/cluster5.png',cluster)
if k == 27:
	cv2.destroyAllWindows() 
elif k == ord('s'):
	cv2.imwrite('Frame_%04d.png'%index,frame2)
	cv2.imwrite('OF_hsv_%04d.png'%index,bgr)


