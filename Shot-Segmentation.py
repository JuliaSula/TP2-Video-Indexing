import matplotlib.pyplot as plt
from matplotlib import ticker
import statsmodels.api
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math



indexV=0
cap = cv2.VideoCapture('../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v')
#cap = cv2.VideoCapture(0)

#Capture de frames
ret, frame1 = cap.read() 
ret, frame2 = cap.read()

#Passage de frames au niveaux de gris
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

#Resize
prvsR=cv2.resize(prvs, (32,32), None, cv2.INTER_AREA)
nextR=cv2.resize(next, (32,32), None, cv2.INTER_AREA)

frameX1=cv2.resize(frame1, (32, 32), None, cv2.INTER_AREA)
frameX2=cv2.resize(frame2, (32,32), None, cv2.INTER_AREA)

#Taille de frames
width =  int(cap.get(3))  
height = int(cap.get(4))

#Objet pour sauvegarder les plans de videos
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
nom='fichier'+str(indexV)
nom = cv2.VideoWriter('firstScene.avi',fourcc,10,(width,height))

print(frame1.shape)
#Frame hsv
hsv = np.zeros_like(frameX1) # Image nulle de même taille que frame1 (affichage OF)
print(hsv.shape)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum



#Conversion Yuv
imYuv = cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV) 
nextimYuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV) 

#Resize
imYuv=cv2.resize(imYuv, (32,32), None, cv2.INTER_AREA)
nextimYuv=cv2.resize(nextimYuv, (32,32), None, cv2.INTER_AREA)


#Variables auxiliaires
index = 1
v_local=[]
v_local.append(1)
index=1
y_acc=[]
v_acc=[]
vt=0
j=0


while(ret):
    index += 1
    
    #Calcul flow
    flow = cv2.calcOpticalFlowFarneback(prvsR,nextR,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire

    #Creation de l'image du flow
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    
    #Histogramme YUV
    #Calcul des histogramme entre frames
    hist= cv2.calcHist([imYuv], [1, 2], None, [32]*2, [0,255, 0,255])
    hist=cv2.normalize(hist, hist)
	
    hist1= cv2.calcHist([nextimYuv], [1, 2], None, [32]*2, [0,255, 0,255])
    hist1=cv2.normalize(hist1, hist1)

    #Calcul de la correlation des histogramme
    y=cv2.compareHist(hist, hist1,cv2.HISTCMP_CORREL)
    y_acc.append(y)
    
   

    #Histogramme Flow
    #Calcul des histogramme entre frames
    histF= cv2.calcHist([flow], [0, 1],None, [32]*2, [-32,32]*2)
   
    #Normalization de l'histogramme
    histF[histF>0]=np.log(histF[histF>0])
    histF=(histF-histF.min())/(histF.max()-histF.min())

    #Fusion entre les valeur des deux histogramme
    alpha=0.5
    v=alpha*y+alpha*(1/(sum(sum(histF))))
    #v=statsmodels.tsa.filters.filtertools.recursive_filter(v, np.ones(10), init=None)
    v_acc.append(v)
    v_local.append(v)

    #index auxiliaire
    j=j+1
    
    #Definition des seuils
    
    #Verification du changement de frame
    m=np.mean(v_local)
    print(abs(v-v_acc[j-2]), v,v_acc[j-2], vt)
    print(indexV)
    if(abs(v-v_acc[j-2])>0.2):
        v_local=[]
        v_local.append(1)
        nom.release()
        indexV=indexV+1
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        nom='fichier'+str(indexV) 
        nom= cv2.VideoWriter('test'+str(indexV)+'.avi',fourcc,10,(width,height)) 
    elif(abs(v-v_acc[j-2])>0.03):
        vt=vt+abs(v)
    if(vt>4):
        vt=0
        v_local.append(1)
        nom.release()
        indexV=indexV+1
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        nom='fichier'+str(indexV) 
        nom= cv2.VideoWriter('test'+str(indexV)+'.avi',fourcc,10,(width,height)) 
        

    #Plots
    #Plot 2D histogramme flow
    """fig, ax =plt.subplots()
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
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)"""
    #Plots
    #Plot 2D variable de fusion
    fig2, ax2 =plt.subplots()	
    ax2 = plt.gca()
    ax2.plot(v_acc)
    plt.xlabel("Index")
    plt.ylabel("Corrélation entre frames")
    plt.draw()
    fig2.canvas.draw()
    #plt.pause(0.05)
    imgt = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    imgt  = imgt.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    imgt = cv2.cvtColor(imgt,cv2.COLOR_RGB2BGR)
    cv2.imshow("Corrélation entre frames",imgt)
   
  
   
    # display image with opencv or any operation you like
    #cv2.imshow("plot",img)
    cv2.imshow("plot1",frame2)
 
    #writing of the video 
    nom.write(frame2)
    
    #dealing with breaks
    result = np.vstack((frameX2,bgr))
    """cv2.imshow('Image et Champ de vitesses (Farnebäck)',result)"""
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break 
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    
    #Continuation du video
    prvs = next
    imYuv=nextimYuv
    ret, frame2 = cap.read()
    frameX2=cv2.resize(frame2, (32,32), None, cv2.INTER_AREA)
    if (ret):
        nextimYuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV) 
        nextimYuv=cv2.resize(nextimYuv, (32,32), None, cv2.INTER_AREA)   
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
        nextR=cv2.resize(next, (32,32), None, cv2.INTER_AREA)
        
cap.release()
nom.release()
cv2.destroyAllWindows()
