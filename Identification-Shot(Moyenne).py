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
def data_sample(y, flow,i):
    y=flow+y
    return y


#Ouverture du flux video
cap = cv2.VideoCapture('../TP2_Videos/firstScene.avi')

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

#Variables Aux
histprvs=[]
hsv = np.zeros_like(frameX1)
hsv[:,:,1] = 255
y=np.zeros((32,32))
histT=np.zeros((32,32))
i=0



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
    
    

    #Calcul Histogramme   
    hist= cv2.calcHist([flow], [0, 1],None, [32]*2, [-32,32]*2)
    hist=(hist-hist.min())/(hist.max()-hist.min())
   

    y=data_sample(y, hist, i)
    i=i+1
  
    
    fig, ax =plt.subplots()
    ax.set_xlim([0, 32 - 1])
    ax.set_ylim([0, 32 - 1])
    ax.set_xlabel('Vx')
    ax.set_ylabel('Vy')
    ax.set_title('2D Color Histogram for Vx  and Vy')
    im = ax.imshow(hist)  
    fig.colorbar(im)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow("plot",img)

    
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

#Calcul moyenne
moy=y/i


#Plot Moyenne
fig, ax =plt.subplots()
ax.set_xlim([0, 32 - 1])
ax.set_ylim([0, 32 - 1])
ax.set_xlabel('Vx')
ax.set_ylabel('Vy')
ax.set_title('2D Color Histogram for Vx  and Vy')
im = ax.imshow(moy)    
fig.colorbar(im)
fig.canvas.draw()
img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imshow("plot",img)
cv2.imwrite('../TP2_Videos/MoyenneR.png', img)

#Definition des quadrants
print(np.mean(moy[0:15, 0:15]), np.mean(moy[16:31, 16:31]), np.mean(moy[0:15, 16:31]),  np.mean(moy[16:31, 0:15]))
#print(y)
n1=np.sum(moy[0:15, 0:15])
n2=np.sum(moy[16:31, 16:31])
n3=np.sum(moy[0:15, 16:31])
n4=np.sum(moy[16:31, 0:15])

#Determination du plan
if(n4>n2 and n3>n1):
    print('gauche')
if(n4<n2 and n3<n1):
    print('droit')
if(n4+n2<n3+n1):
    print('down')
if(n4+n2>n3+n1):
    print('up')




