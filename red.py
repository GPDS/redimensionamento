import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from os import system
from skimage.metrics import structural_similarity as ssim
#from numba import njit

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
pasta = "TRAIN/"

for i in range(35100,n):
    if i % 100 == 0:
    	p = np.round(i/n*100,2)
    	print("Total concluído: " + f"{p:.2f}" + "%")
    path = pasta + data_frame.image[i] + ".jpeg"

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

    imagem2 = imagem[coord3:coord4,coord1:coord2,:]
    # cv2.imshow('Image',imagem2)
    # cv2.waitKey(0)

    # print(f"x1 = {coord1}, x2 = {len(imagem[0,:,0])-coord2}, y1 = {coord3}, y2 = {len(imagem[:,0,0])-coord4}")

    imagem3 = cv2.resize(imagem2, (224,224), interpolation=cv2.INTER_LANCZOS4)
    # cv2.imshow('Image',imagem3)
    # cv2.waitKey(0)

    npath = "redim" + pasta + path[len(pasta):]
    # print(npath)

    cv2.imwrite(npath, imagem3)

    # gray_image = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)
    # gray_image2 = cv2.cvtColor(imagem3, cv2.COLOR_BGR2GRAY)

    # score1 = Inf_spatial(gray_image)
    # score2 = Inf_spatial(gray_image2)
    # print(f"Imagem 1 = {score1}, Imagem 2 = {score2}")
