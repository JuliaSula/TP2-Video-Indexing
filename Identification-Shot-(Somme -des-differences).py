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

#Fonction pour obtenir les difference entre histogrammes
def data_sample(y, hist, histPrv, index):
    if(index>3):
        y.append(hist-histPrv)
    return hist

#Ouverture du flux video
cap = cv2.VideoCapture('../TP2_Videos/testLATERAL2.webm')

#Capture des frames
index = 1
ret, frame1 = cap.read() # Passe à l'image suivante
ret, frame2 = cap.read()

# Passage en niveaux de gris
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) 
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

#Resize: reduire la resolution spatiale cree des histogrammes plus robustes et moins sensibles a mouvmenets
prvsR=cv2.resize(prvs, (32,32), None, cv2.INTER_AREA)
nextR=cv2.resize(next, (32,32), None, cv2.INTER_AREA)
frameX1=cv2.resize(frame1, (32,32), None, cv2.INTER_AREA)
frameX2=cv2.resize(frame2, (32,32), None, cv2.INTER_AREA)

#Creation des variables auxiliares
histPrv=[]
hsv = np.zeros_like(frameX1)
hsv[:,:,1] = 255
y=[]
histT=[]
i=0


#Loop Principal
while(ret):
    index += 1
    #Calcul du flow
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
    
    #Calcul du histogramme
    hist= cv2.calcHist([flow], [0, 1],None, [32]*2, [-32,32]*2)
    hist=(hist-hist.min())/(hist.max()-hist.min())
    i=i+1
 
    #Calcul des difference
    histPrv=data_sample(y, hist, histPrv,i)
    
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
#Somme des difference
plt.imshow(sum(y[:]))
s=sum(y[:])

#Determination des valeurs de la somme de difference qui seront utilises
l=np.where(s>0.2)

#Dans le cas ou l est empty, le plan est statique
if(not l):
    print('statique')
else:
    #Calculant les distance du pixel au centre
    centre= 16*np.ones(np.shape(l))
    diff=l-centre
    w=diff*s[l]
    #Calculant les modes de la distance
    print(stats.mode(diff[0,:]),stats.mode(diff[1,:]))
    a=stats.mode(diff[0,:])[0] # distance en x
    b=stats.mode(diff[1,:])[0] # distance en y
    #Determinant le type de plan
    if(a>0):
        print('travelling right')
    elif (a==0):
        print('')
    else:
        print('travelling left')
 
    if(b>0):
        print('travelling up')
    elif (b==0):
        print('-')
    else:
        print('travelling down')
plt.show()




