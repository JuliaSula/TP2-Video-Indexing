import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import statistics
from scipy import stats
from scipy.signal import find_peaks
from scipy.integrate import simps
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage as ndi
from sklearn.decomposition import FastICA


#Ouverture du flux video
cap = cv2.VideoCapture('../TP2_Videos/testLATERAL.webm')

#Frames
index = 1
ret, frame1 = cap.read() # Passe à l'image suivante
ret, frame2 = cap.read()

# Passage en niveaux de gris
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) 
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

#Resize
prvsR=cv2.resize(prvs, (32,32), None, cv2.INTER_AREA)
nextR=cv2.resize(next, (32,32), None, cv2.INTER_AREA)
frameX1=cv2.resize(frame1, (32,32), None, cv2.INTER_AREA)
frameX2=cv2.resize(frame2, (32,32), None, cv2.INTER_AREA)

#Variables aux
histPrv=[]
hsv = np.zeros_like(frameX1)
hsv[:,:,1] = 255
y=np.zeros((32,32))
flowT=np.zeros((32,32,2))
histT=[]
i=0

histY=[]

while(ret):
    index += 1
    
    flow = cv2.calcOpticalFlowFarneback(prvsR,nextR,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    

    #Calcul hist
    hist= cv2.calcHist([flow], [0, 1],None, [32]*2, [-32,32]*2)   
    hist=(hist-hist.min())/(hist.max()-hist.min())
    
    #Matrix de histogrammes
    i=i+1
    y=y+hist
    histT.append(hist)
    histY.append(np.transpose(hist))
   
    #Show
    result = np.vstack((frameX2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',result)
    cv2.imshow('test', frame2)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    prvs = next
    ret, frame2 = cap.read()
    #frame2=cv2.imread('../TP2_Videos/4.jpg')
    if (ret):
        frameX2=cv2.resize(frame2, (32,32), None, cv2.INTER_AREA)
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
        nextR=cv2.resize(next, (32,32), None, cv2.INTER_AREA)
cap.release()
cv2.destroyAllWindows()
v=[]



inte=simps(histY, x=None, dx=1, axis=-1, even='avg')
plt.imshow(np.transpose(inte))
plt.show()
diffy=((inte[0]-inte[np.shape(inte)[0]-1]))

inte=simps(histT, x=None, dx=1, axis=-1, even='avg')
plt.imshow(inte)
plt.show()
print(np.shape(inte[0]))
diffx=((inte[0]-inte[np.shape(inte)[0]-1]))

print(diffx, diffy)
if(sum(abs(diffx[0:15]))>sum(abs(diffx[16:31]))):
    print('left')
else:
    print('right')
if(sum(abs(diffy[0:15]))>sum(abs(diffy[16:31]))):
    print('up')
else:
    print('down')
print(np.where(diffx>0))
print(np.where(diffy>0))

"""
plt.imshow(flowT[:,:,1])
plt.show()
print(flowT)
ica = FastICA(n_components=1)
S_ = ica.fit_transform(flowT[:,:,0])
plt.imshow(S_)
plt.show()
emc2_image_ica = ica.fit_transform(flowT[:,:,1])
emc2_restored = ica.inverse_transform(emc2_image_ica)
 
# show image to screen
plt.imshow(emc2_restored)
plt.show()"""




