import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from os import system
#from skimage.metrics import structural_similarity as ssim

def crop_image_from_gray(img,tol=7):
    if img.ndim == 2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img, sigmaX = 30, contrast = 4, center = 128):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape    
    
    x = width // 2
    y = height // 2
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img, contrast, cv2.GaussianBlur(img, (0,0), sigmaX), -contrast, center)
    return img



data_frame = pd.read_csv("aptos2019/train_1.csv")
#data_frame = data_frame.drop(columns="level")
n = data_frame.shape[0]
type = "train"
pasta = f"aptos2019/{type}_images/"
res = 600

for i in range(0,n):
    if i % (n // 20) == 0:
        p = np.round(i/n*100,2)
        print("Total concluído: " + f"{p:.2f}" + "%")
    path = pasta + data_frame.id_code[i] + ".png"
    #print(path)
    try:
        imagem = cv2.imread(path)

        imagem = cv2.resize(circle_crop(imagem), (res,res), interpolation=cv2.INTER_LANCZOS4)

        npath = f"aptos2019_redim_{res}x{res}/{type}_images/" + path[len(pasta):]

        cv2.imwrite(npath, imagem)
    except:
        print("Gorou o ovo")
