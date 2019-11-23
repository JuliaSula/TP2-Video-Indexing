import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from skimage.measure import compare_ssim
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
from PIL import Image



#Ouverture du flux video
cap = cv2.VideoCapture('../TP2_Videos/Right.webm')
#'../TP2_Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v'

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
    
    
    #Calcul histogramme
    hist= cv2.calcHist([flow], [0, 1],None, [32]*2, [-32,32]*2)
    hist=(hist-hist.min())/(hist.max()-hist.min())
    i=i+1
   
    
    #Creation du vecteur de flow   
    histT.append(flow[:,:,0])
    histY.append(flow[:,:,1])
    #Showing
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
    if (ret):
        frameX2=cv2.resize(frame2, (32,32), None, cv2.INTER_AREA)
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
        nextR=cv2.resize(next, (32,32), None, cv2.INTER_AREA)
cap.release()
cv2.destroyAllWindows()
v=[]

#Calcul Integral
inte=simps(histT, x=None, dx=1, axis=-1, even='avg')
plt.imshow(inte)
plt.show()
x=np.mean(inte)
inte=simps(histY, x=None, dx=1, axis=-1, even='avg')
plt.imshow(inte)
plt.show()
y=np.mean(inte)

print(x,y)
#Determination du plan
if(abs(y)>5*abs(x)):
    if(y>0):
        print('travelling up')
    if(y<0):
        print('travelling down')
elif(abs(y)*5<abs(x)):
    if(x>0):
        print('travelling left')
    if(x<0):
        print('travelling right')
elif(y-x<2):
    print('other mouvement')
else:
    print('statique')
plt.show()




