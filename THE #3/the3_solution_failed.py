# onur yaman - 2007961
# ilkim oğul - 2237675
import os
import cv2 as cv
import math
from skimage import io, data, transform
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import array


INPUT_PATH = "./THE3_Images/"
OUTPUT_PATH = "./Outputs/"

def read_image(img_path, rgb=True):
    img = io.imread(img_path)
    return img

def write_image(img_list, output_path):
    plt.imshow(img)
    plt.savefig(output_path)

def detect_faces(img,lr,lg,lb):
    ci=[[],[],[],[],[]]
    cj=[[],[],[],[],[]]
    leng=[0,0,0,0,0]
    row = img.shape[0]
    col = img.shape[1]
    skin1=[]
    skin2=[]
    norm = np.zeros((row,col))
    norm = cv.normalize(img,  norm, 0, 255, cv.NORM_MINMAX)
    ms = np.random.randint(255, size=(5, 3))
    ms2=np.zeros((5,3))
    i=0
    j=0
    mi=999
    asi=0
    u=0
    s=0
    a=0
    me=0
    while s!=1:
        leng=[0,0,0,0,0]
        ci=[[],[],[],[],[]]
        cj=[[],[],[],[],[]]
        i=0
        while i<row:
            j=0
            while j<col:
                me=0
                mi=999
                while me<5:
                    d=math.sqrt((ms[me][0]-norm[:,:,0][i][j])**2+(ms[me][1]-norm[:,:,1][i][j])**2+(ms[me][2]-norm[:,:,2][i][j])**2)
                    if d<mi:
                        mi=d
                        asi=me
                    me=me+1
                ci[asi].append(i)
                cj[asi].append(j)
                leng[asi]=leng[asi]+1
                j=j+1
            i=i+1
        j=0
        ms=np.zeros((5,3), dtype=int)
        while j<3:
            i=0
            while i<5:
                u=0
                if leng[i]!=0:
                    while u<leng[i]:
                        ms[i][j]+=norm[ci[i][u]][cj[i][u]][j]
                        u+=1
                    ms[i][j]=ms[i][j]/leng[i]
                i+=1
            j+=1

        if np.array_equal(ms,ms2) and s==0:
            s+=1
        elif np.array_equal(ms,ms2) and s==1:
            s+=1
        else:
            ms2=ms.copy()
            s=0
    i=0
    j=0

    while j<4:
        i=0
        mi=999
        while i<5:
            d=math.sqrt((lr[j]-ms[i][0])**2+(lg[j]-ms[i][1])**2+(lb[j]-ms[i][2])**2)
            if(d<mi):
                mi=d
                asi=j
                a=i
            i+=1
        if asi==1 or asi==2:
            u=0
            while u<leng[a]:
                img[ci[a][u]][cj[a][u]]=[255,0,0]
                u+=1
        j+=1
    return img
#nonface=

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #PART1
label1r=[157,253,210,156]
label1g=[118,210,100,158]
label1b=[136,190,50,86]

label2r=[230,220,170,65]
label2g=[211,180,110,52]
label2b=[168,160,110,69]

label3r=[255,195,135,18]
label3g=[255,149,90,20]
label3b=[255,131,67,17]



img = read_image(INPUT_PATH + "2_source.png")
out=detect_faces(img,label1r,label1g,label1b)
write_image(out, OUTPUT_PATH + "2.png")


    # BONUS
    # Define the following function
    # equalized = adaptive_histogram_equalization(img)
    # extract_save_histogram(equalized, OUTPUT_PATH + "adaptive_equalized_histogram.png")
    # write_image(output, OUTPUT_PATH + "adaptive_enhanced_image.png")
