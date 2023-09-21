import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from os import system
#from skimage.metrics import structural_similarity as ssim
from numba import njit

@njit
def forROW(rowsf,collumsf,ponto_inicialf,new_imagef,imagem2):
    for cl in range(collumsf):
        for rw in range(ponto_inicialf,ponto_inicialf+rowsf):
            for camada in range(3):
                new_imagef[rw,cl,camada] = imagem2[rw-ponto_inicialf,cl,camada]
    return new_imagef

@njit
def forCOLLUM(rowsf,collumsf,ponto_inicialf,new_imagef,imagem2):
    for cl in range(ponto_inicialf,ponto_inicialf+collumsf):
        for rw in range(rowsf):
            for camada in range(3):
                new_imagef[rw,cl,camada] = imagem2[rw,cl-ponto_inicialf,camada]
    return new_imagef

# system("clear")

def Inf_spatial(img_grayscale):
    sh = cv2.Sobel (img_grayscale, cv2.CV_64F , 1, 0, ksize =1)
    sv = cv2.Sobel (img_grayscale , cv2.CV_64F , 0, 1, ksize =1)

    #SIr = np.sqrt (sh **2 + sv **2)
    SIr = np.sqrt(np.square(sh) + np.square(sv))

    SI_mean = np.sum(SIr ) / (SIr. shape [0] * SIr . shape [1])
    SI_rms = np.sqrt (np.sum (SIr **2) / (SIr. shape [0] * SIr . shape [1]) )
    SI_stdev = np.sqrt (np.sum(SIr **2 - SI_mean **2) / (SIr. shape [0] *SIr. shape [1]) )

    return SI_stdev

data_frame = pd.read_csv("trainLabels.csv")
data_frame = data_frame.drop(columns="level")

# print(data_frame.image[2])

n = 35126
inicio = 26161
pasta = "TRAIN/"

for ii in range(inicio,n):
    if ii % 100 == 0:
        p = np.round(ii/n*100,2)
        print("Total concluído: " + f"{p:.2f}" + "%")
    path = pasta + data_frame.image[ii] + ".jpeg"

    imagem = cv2.imread(path)

    threshold = 0.01

    #Busca X
    flag = 1
    x = np.zeros(len(imagem[0,:,0]))
    for i in range(len(imagem[0,:,0])):
        k = np.sum(imagem[:,i,:])
        x[i] = k
    x = x/np.max(x)

    #Busca Y
    flag = 1
    y = np.zeros(len(imagem[:,0,0]))
    for i in range(len(imagem[:,0,0])):
        k = np.sum(imagem[i,:,:])
        y[i] = k
    y = y/np.max(y)

    #x
    flag = 1
    for i in range(len(x)):
        if x[i] > threshold and flag == 1:
            flag = 0
            coord1 = i

    flag = 1
    for i in range(len(x)):
        if x[len(x)-i-1] > threshold and flag == 1:
            flag = 0
            coord2 = len(x)-i
            #imagem[:,coord2,:] = 255

    #y
    flag = 1
    for i in range(len(y)):
        if y[i] > threshold and flag == 1:
            flag = 0
            coord3 = i

    flag = 1
    for i in range(len(y)):
        if y[len(y)-i-1] > threshold and flag == 1:
            flag = 0
            coord4 = len(y)-i

    ###################################################################

    #plt.plot(x)
    #plt.show()
    rows = coord4 - coord3
    collums = coord2 - coord1
    
    imagem2 = imagem[coord3:coord4,coord1:coord2,:]
	
    if collums > rows:
        new_image = np.zeros((collums,collums,3), dtype="uint8")
        ponto_inicial = int(collums/2 - rows/2)
        new_image = forROW(rows,collums,ponto_inicial,new_image,imagem2)
    elif rows > collums:
        new_image = np.zeros((rows,rows,3), dtype="uint8")
        ponto_inicial = int(rows/2 - collums/2)
        new_image = forCOLLUM(rows,collums,ponto_inicial,new_image,imagem2)
    else:
        new_image = imagem2
	
    imagem3 = cv2.resize(new_image, (224,224), interpolation=cv2.INTER_LANCZOS4)
    npath = "redim_teste" + pasta + path[len(pasta):]
    cv2.imwrite(npath, imagem3)
