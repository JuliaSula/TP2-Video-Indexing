import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math


#Ouverture du flux video
cap = cv2.VideoCapture(0)
#'../TP2_Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v'

ret, frame1 = cap.read() # Passe à l'image suivante

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

while(ret):
    index += 1
    
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
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
    bins=10
    print(frame1.shape, flow.shape)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    
    #Calcul et normalization du histogramme
    hist= cv2.calcHist([flow], [0, 1],None, [64]*2, [-64,64]*2)
    hist[hist>0]=np.log(hist[hist>0])
    hist=(hist-hist.min())/(hist.max()-hist.min())
    
    #Configuration et show histogramme
    fig, ax =plt.subplots()
    ax.set_xlim([0, 64 - 1])
    ax.set_ylim([0, 64 - 1])
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
    # display image with opencv or any operation you like
    cv2.imshow("plot",img)
    
  
    result = np.vstack((frame2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',result)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cap.release()
cv2.destroyAllWindows()
