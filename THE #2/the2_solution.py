# onur yaman - 2007961
# ilkim oğul - 2237675


import os
import cv2
import math
from skimage import io, data, transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy import linalg
from PIL import Image


INPUT_PATH = "./THE_2 Images/"
OUTPUT_PATH = "./Outputs/"


def read_image(img_path, rgb=True):
    img = io.imread(img_path)
    return img

def write_image(img_list, output_path):
    row = img_list[0].shape[0]
    col = img_list[0].shape[1]
    arr=np.zeros((row,col,3),np.float32)
    fin=np.zeros((row,col,3),np.uint8)
    sum=img_list[0]+img_list[1]+img_list[2]
    arr[:,:,0]=(img_list[0])/sum*255.0
    arr[:,:,1]=(img_list[1])/sum*255.0
    arr[:,:,2]=(img_list[2])/sum*255.0
    fin=cv2.convertScaleAbs(arr)
    plt.imshow(fin)
    plt.savefig(output_path)

def fourier_transform(img,out):
    fig, ax = plt.subplots(3)
    for i in range(3):
        fig = np.log(1+fp.fftshift(abs(fp.fft2(img[:, :,i]))))
        ax[i].imshow(fig,cmap='gray')

    plt.savefig(out)
    plt.close()

def cosine_transform(img,out):
    fig, ax = plt.subplots(3)
    for i in range(3):
        fig = fp.dctn(img[:, :, i])
        ax[i].imshow(fig,cmap='gray')
    plt.savefig(out)
    plt.close()

def hadamard_transform(img,out):
    fig, ax = plt.subplots(3)
    w = int(np.shape(img)[1])
    h = int(np.shape(img)[0])
    iarr=[]
    had = [[]]
    u=1

    while u<w and u<h:
        u=2*u

    u=int(u/2)
    had=linalg.hadamard(u)
    hadt=np.transpose(had)
    a=(w-u)/2
    a=int(a)
    b=(h-u)/2
    b=int(b)
    for i in range(3):
        iarr=img[:,:,i]
        iarr=iarr[b:(b+u),a:(a+u)]
        ax[i].imshow(np.matmul(np.matmul(had, iarr), hadt),cmap='gray')
    plt.savefig(out)
    plt.close()

def applyFilter(img, H):
    row = img.shape[0]
    col = img.shape[1]
    matrix = np.zeros((row, col))
    n = 1
    for i in range(row):
        for j in range(col):
            matrix[i, j] = n
            n = -n
        n = -n
    return np.real(ifftshift(ifft2(fft2(fftshift(img*matrix))*H))*matrix)

def idealFilter(img, cutoff_frequency, lowpass=True):
    row = img.shape[0]
    col = img.shape[1]
    rowsIndex = np.zeros((row, col))
    colsIndex = np.zeros((row, col))
    if row == col :
        for i in range(row):
            for j in range(col):
                rowsIndex[i, j] = i
        colsIndex = rowsIndex.T
    else:
        for i in range(row):
            for j in range(col):
                rowsIndex[i, j] = i
                colsIndex[i, j] = j
    blank = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            blank[i, j] = np.sqrt(pow((rowsIndex[i, j]-row//2), 2) + pow((colsIndex[i, j]-col//2), 2))
    if lowpass:
        L = np.heaviside(-1*(blank-cutoff_frequency), 1)
        return L
    else:
        H = np.heaviside(1*(blank-cutoff_frequency), 1)
        return H

def gaussianFilter(img, cutoff_frequency, lowpass=True):
    row = img.shape[0]
    col = img.shape[1]
    rowsIndex = np.zeros((row, col))
    colsIndex = np.zeros((row, col))
    if row == col:
        for i in range(row):
            for j in range(col):
                rowsIndex[i, j] = i
        colsIndex = rowsIndex.T
    else:
        for i in range(row):
            for j in range(col):
                rowsIndex[i, j] = i
                colsIndex[i, j] = j
    blank = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            blank[i, j]=np.sqrt(pow((rowsIndex[i, j]-row//2), 2)+pow((colsIndex[i, j]-col//2), 2))
    if lowpass:
        L = pow(math.e, (-1*(pow((blank / (2 * cutoff_frequency)), 2))))
        return L
    else:
        H = 1-pow(math.e, (-1*(pow((blank / (2 * cutoff_frequency)), 2))))
        return H

def butterWorhFilter(img, cutoff_frequency, n ,lowpass=True):
    row = img.shape[0]
    col = img.shape[1]
    rowsIndex=np.zeros((row, col))
    colsIndex=np.zeros((row, col))
    if row == col :
        for i in range(row):
            for j in range(col):
                rowsIndex[i, j] = i
        colsIndex = rowsIndex.T
    else:
        for i in range(row):
            for j in range(col):
                rowsIndex[i, j] = i
                colsIndex[i, j] = j
    blank = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            blank[i, j] = np.sqrt(pow((rowsIndex[i, j]-row//2), 2)+pow((colsIndex[i, j]-col//2), 2))
    if lowpass:
        L = 1/(1+(blank/cutoff_frequency)**(2*n))
        return L
    else:
        blank = np.heaviside(blank,0.1)-np.sign(D)+D
        H = 1/(1+(cutoff_frequency/D)**(2*n))
        return H

def low_pass_filtering(img, type, cutoff_frequency, channel):
    img = img[:, :, channel]
    if type == "Ideal":
        L = idealFilter(img, cutoff_frequency)
        ILPFImg = applyFilter(img, L)
        return ILPFImg
    elif type == "Gaussian":
        L = gaussianFilter(img, cutoff_frequency)
        GLPFImg = applyFilter(img, L)
        return GLPFImg
    elif type == "Butterworh":
        L = butterWorhFilter(img, cutoff_frequency, n=2)
        BLPFImg = applyFilter(img, L)
        return BLPFImg

def high_pass_filtering(img, type, cutoff_frequency, channel):
    img = img[:, :, channel]
    if type == "Ideal":
        H = idealFilter(img, cutoff_frequency, lowpass=False)
        IHPFImg = applyFilter(img, H)
        return IHPFImg
    elif type == "Gaussian":
        H = gaussianFilter(img, cutoff_frequency, lowpass=False)
        GHPFImg = applyFilter(img, H)
        return GHPFImg
    elif type == "Butterworh":
        H = butterWorhFilter(img, cutoff_frequency, n=2, lowpass=False)
        BHPFImg = applyFilter(img, H)
        return BHPFImg

def band_filtering(img, type, channel):
    out=low_pass_filtering(img,type,10,channel)
    out=high_pass_filtering(out,type,30,channel)

    if type == "Reject":
        out=img-out
    return out

def enhance_image(img):
    return img



if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # STEP2 - STEP1 INCLUDED - 6 outputs
    img = read_image(INPUT_PATH + "1.png")
    output = fourier_transform(img, OUTPUT_PATH + "F1.png")

    img = read_image(INPUT_PATH + "1.png")
    output = hadamard_transform(img, OUTPUT_PATH +"H1.png")

    img = read_image(INPUT_PATH + "1.png")
    output = cosine_transform(img, OUTPUT_PATH +"C1.png")

    img = read_image(INPUT_PATH + "2.png")
    output = fourier_transform(img, OUTPUT_PATH +"F2.png")

    img = read_image(INPUT_PATH + "2.png")
    output = hadamard_transform(img, OUTPUT_PATH +"H2.png")

    img = read_image(INPUT_PATH + "2.png")
    output = cosine_transform(img, OUTPUT_PATH +"C2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Ideal", cutoff_frequency=15, channel=0)
output_g = low_pass_filtering(img, type="Ideal", cutoff_frequency=15, channel=1)
output_b = low_pass_filtering(img, type="Ideal", cutoff_frequency=15, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "ILP_r1.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Ideal", cutoff_frequency=30, channel=0)
output_g = low_pass_filtering(img, type="Ideal", cutoff_frequency=30, channel=1)
output_b = low_pass_filtering(img, type="Ideal", cutoff_frequency=30, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "ILP_r2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Ideal", cutoff_frequency=80, channel=0)
output_g = low_pass_filtering(img, type="Ideal", cutoff_frequency=80, channel=1)
output_b = low_pass_filtering(img, type="Ideal", cutoff_frequency=80, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "ILP_r3.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Gaussian", cutoff_frequency=15, channel=0)
output_g = low_pass_filtering(img, type="Gaussian", cutoff_frequency=15, channel=1)
output_b = low_pass_filtering(img, type="Gaussian", cutoff_frequency=15, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "GLP_r1.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Gaussian", cutoff_frequency=30, channel=0)
output_g = low_pass_filtering(img, type="Gaussian", cutoff_frequency=30, channel=1)
output_b = low_pass_filtering(img, type="Gaussian", cutoff_frequency=30, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "GLP_r2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Gaussian", cutoff_frequency=80, channel=0)
output_g = low_pass_filtering(img, type="Gaussian", cutoff_frequency=80, channel=1)
output_b = low_pass_filtering(img, type="Gaussian", cutoff_frequency=80, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "GLP_r3.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Butterworh", cutoff_frequency=15, channel=0)
output_g = low_pass_filtering(img, type="Butterworh", cutoff_frequency=15, channel=1)
output_b = low_pass_filtering(img, type="Butterworh", cutoff_frequency=15, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BLP_r1.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Butterworh", cutoff_frequency=30, channel=0)
output_g = low_pass_filtering(img, type="Butterworh", cutoff_frequency=30, channel=1)
output_b = low_pass_filtering(img, type="Butterworh", cutoff_frequency=30, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BLP_r2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = low_pass_filtering(img, type="Butterworh", cutoff_frequency=80, channel=0)
output_g = low_pass_filtering(img, type="Butterworh", cutoff_frequency=80, channel=1)
output_b = low_pass_filtering(img, type="Butterworh", cutoff_frequency=80, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BLP_r3.png")

# STEP4 - STEP1 INCLUDED - 9 outputs
img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Ideal", cutoff_frequency=15, channel=0)
output_g = high_pass_filtering(img, type="Ideal", cutoff_frequency=15, channel=1)
output_b = high_pass_filtering(img, type="Ideal", cutoff_frequency=15, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "IHP_r1.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Ideal", cutoff_frequency=30, channel=0)
output_g = high_pass_filtering(img, type="Ideal", cutoff_frequency=30, channel=1)
output_b = high_pass_filtering(img, type="Ideal", cutoff_frequency=30, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "IHP_r2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Ideal", cutoff_frequency=80, channel=0)
output_g = high_pass_filtering(img, type="Ideal", cutoff_frequency=80, channel=1)
output_b = high_pass_filtering(img, type="Ideal", cutoff_frequency=80, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "IHP_r3.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Gaussian", cutoff_frequency=15, channel=0)
output_g = high_pass_filtering(img, type="Gaussian", cutoff_frequency=15, channel=1)
output_b = high_pass_filtering(img, type="Gaussian", cutoff_frequency=15, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "GHP_r1.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Gaussian", cutoff_frequency=30, channel=0)
output_g = high_pass_filtering(img, type="Gaussian", cutoff_frequency=30, channel=1)
output_b = high_pass_filtering(img, type="Gaussian", cutoff_frequency=30, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "GHP_r2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Gaussian", cutoff_frequency=80, channel=0)
output_g = high_pass_filtering(img, type="Gaussian", cutoff_frequency=80, channel=1)
output_b = high_pass_filtering(img, type="Gaussian", cutoff_frequency=80, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "GHP_r3.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Butterworh", cutoff_frequency=15, channel=0)
output_g = high_pass_filtering(img, type="Butterworh", cutoff_frequency=15, channel=1)
output_b = high_pass_filtering(img, type="Butterworh", cutoff_frequency=15, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BHP_r1.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Butterworh", cutoff_frequency=30, channel=0)
output_g = high_pass_filtering(img, type="Butterworh", cutoff_frequency=30, channel=1)
output_b = high_pass_filtering(img, type="Butterworh", cutoff_frequency=30, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BHP_r2.png")

img = read_image(INPUT_PATH + "3.png")
output_r = high_pass_filtering(img, type="Butterworh", cutoff_frequency=80, channel=0)
output_g = high_pass_filtering(img, type="Butterworh", cutoff_frequency=80, channel=1)
output_b = high_pass_filtering(img, type="Butterworh", cutoff_frequency=80, channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BHP_r3.png")

# STEP5 - STEP1 INCLUDED - 8 outputs
img = read_image(INPUT_PATH + "4.png")
output_r = band_filtering(img, type="Reject", channel=0)
output_g = band_filtering(img, type="Reject", channel=1)
output_b = band_filtering(img, type="Reject", channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BR1.png")

img = read_image(INPUT_PATH + "5.png")
output_r = band_filtering(img, type="Reject", channel=0)
output_g = band_filtering(img, type="Reject", channel=1)
output_b = band_filtering(img, type="Reject", channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BR2.png")

img = read_image(INPUT_PATH + "4.png")
output_r = band_filtering(img, type="Pass", channel=0)
output_g = band_filtering(img, type="Pass", channel=1)
output_b = band_filtering(img, type="Pass", channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BP1.png")

img = read_image(INPUT_PATH + "5.png")
output_r = band_filtering(img, type="Pass", channel=0)
output_g = band_filtering(img, type="Pass", channel=1)
output_b = band_filtering(img, type="Pass", channel=2)
output_list = []
output_list.append(output_r)
output_list.append(output_g)
output_list.append(output_b)
write_image(output_list, OUTPUT_PATH + "BP2.png")

# STEP6 - STEP1 INCLUDED - 2 outputs - enchance_image() function may need other parameters
img = read_image(INPUT_PATH + "6.png")
output = enhance_image(img)
write_image(output, OUTPUT_PATH + "Space6.png")

img = read_image(INPUT_PATH + "7.png")
output = enhance_image(img)
write_image(output, OUTPUT_PATH + "Space7.png")
