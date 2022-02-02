# onur yaman - 2007961
# ilkim oğul - 2237675
import os
from skimage import io, feature, color,filters,segmentation
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import square, disk
from skimage.filters.rank import median
from skimage.transform import rescale
import cv2 as cv
from sklearn import cluster
from skimage.future import graph

INPUT_PATH = "./THE4_Images/"
OUTPUT_PATH = "./Outputs/"

def read_image(img_path, rgb = True):
    img = io.imread(img_path)
    return img

def write_image(img, output_path, rgb = True):
    plt.imshow(img,cmap='gray')
    plt.savefig(output_path)

def count1 (img,a,b,c):
    thresh = cv.threshold(img, a, 255, cv.THRESH_BINARY)[1]
    gray = color.rgb2gray(thresh)
    bin = np.uint8((gray> 0.1) * 255)
    k1 = np.ones((b,b),np.uint8)
    k2 = np.ones((35,35),np.uint8)
    erosion = cv.erode(bin,k1,iterations = 1)
    dilation = cv.dilate(erosion,k2,iterations = c)
    count, _ = cv.connectedComponents(dilation)
    print("The number of flowers in image A1 is", count - 1)
    return dilation


def count2 (img,a,b,c):
    thresh = cv.threshold(img, a, 255, cv.THRESH_BINARY)[1]
    gray = color.rgb2gray(thresh)
    bin = np.uint8((gray> 0.1) * 255)
    k1 = np.ones((b,b),np.uint8)
    k2 = np.ones((35,35),np.uint8)
    erosion = cv.erode(bin,k1,iterations = 1)
    dilation = cv.dilate(erosion,k2,iterations = c)
    count, _ = cv.connectedComponents(dilation)
    print("The number of flowers in image A2 is", count - 1)
    return dilation

def count3 (img,a,b,c):
    row = img.shape[0]
    col = img.shape[1]
    thresh = cv.threshold(img, a, 255, cv.THRESH_BINARY)[1]
    gray = color.rgb2gray(thresh)
    i=0
    while i<row:
        j=0
        while j<col:
            if gray[i][j]>0.8:
                gray[i][j]=0
            j=j+1
        i=i+1

    bin = np.uint8((gray> 0.1) * 255)
    k1 = np.ones((b,b),np.uint8)
    k2 = np.ones((35,35),np.uint8)
    erosion = cv.erode(bin,k1,iterations = 1)
    dilation = cv.dilate(erosion,k2,iterations = c)
    count, _ = cv.connectedComponents(dilation)
    print("The number of flowers in image A3 is", count - 1)
    return dilation


def write_images(imgs, output_path, rgb=False):
    fig, ax = plt.subplots(5)
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
    ax[0].imshow(imgs[0])
    ax[1].imshow(imgs[1])
    ax[2].imshow(imgs[2])
    ax[3].imshow(imgs[3])
    ax[4].imshow(imgs[4])
    plt.savefig(output_path)
    plt.close()

def segmentation_mean_shift_for_others(img, param1, param2):
    imgs = []
    original_shape = img.shape
    img_flatted = np.reshape(img, [-1, 3])
    bandwidth = cluster.estimate_bandwidth(img_flatted, quantile=param1, n_samples=param2)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(img_flatted)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    img_segmented = np.reshape(labels, original_shape[:2])
    superpixels = color.label2rgb(img_segmented, img, kind='avg')
    imgs.append(superpixels)

    b, g, r = cv.split(superpixels)
    b_edge = cv.Canny(b, 255, 0)
    g_edge = cv.Canny(g, 255, 0)
    r_edge = cv.Canny(r, 255, 0)
    edge = cv.merge([b_edge, g_edge, r_edge])
    imgs.append(edge)

    return imgs

def segmentation_mean_shift_for_B3(img, param1, param2):
    imgs = []
    original_shape = img.shape
    img_flatted = np.reshape(img, [-1, 3])
    bandwidth = cluster.estimate_bandwidth(img_flatted, quantile=param1, n_samples=param2)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(img_flatted)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    img_segmented = np.reshape(labels, original_shape[:2])
    superpixels = color.label2rgb(img_segmented, img, kind='avg')
    imgs.append(superpixels)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    img_opened = cv.morphologyEx(superpixels, cv.MORPH_OPEN, kernel)
    result = cv.morphologyEx(img_opened, cv.MORPH_CLOSE, kernel).astype(np.uint8)
    result_g = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    result_b = cv.threshold(result_g, 100, 255, cv.THRESH_BINARY)[1]
    result_b = cv.cvtColor(result_b, cv.COLOR_GRAY2BGR)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gradient = cv.morphologyEx(result_b, cv.MORPH_GRADIENT, kernel)
    gradient = (gradient * np.array([0, 255, 0]) / 255).astype(np.uint8)
    boundry_overlay = combine_imgs(img, gradient.astype(np.uint8))
    imgs.append(boundry_overlay)
    return imgs

def combine_imgs(original, gradient):
    gray = cv.cvtColor(gradient, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    original_bg = cv.bitwise_and(original, original, mask=mask_inv)
    gradient_fg = cv.bitwise_and(gradient, gradient, mask=mask)
    result = cv.add(original_bg, gradient_fg)
    return result

def segmentation_n_cut_for_B3(img, param1, param2):
    imgs = []
    rows, columns = img.shape[0], img.shape[1]
    img_gb = np.concatenate((img[:, :, :2], np.zeros((rows, columns, 1))), axis=2)
    labels1 = segmentation.slic(img_gb, compactness=param1, n_segments=param2)
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    result = color.label2rgb(labels2, img, kind='avg', bg_label=0).astype(np.uint8)
    imgs.append(result)
    result_g = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    result_b = cv.threshold(result_g, 131, 255, cv.THRESH_BINARY)[1]
    result_b = cv.cvtColor(result_b, cv.COLOR_GRAY2BGR)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gradient = cv.morphologyEx(result_b, cv.MORPH_GRADIENT, kernel)
    gradient = (gradient * np.array([0, 255, 0]) / 255).astype(np.uint8)
    output = combine_imgs(img.astype(np.uint8), gradient.astype(np.uint8))
    imgs.append(output)
    return imgs

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    img = read_image(INPUT_PATH + "A1.png")
    output = count1(img,110,40,1)
    write_image(output, OUTPUT_PATH + "A1.png")

    img = read_image(INPUT_PATH + "A2.png")
    output = count2(img,160,22,6)
    write_image(output, OUTPUT_PATH + "A2.png")

    img = read_image(INPUT_PATH + "A3.png")
    output = count3(img,200,30,5)
    write_image(output, OUTPUT_PATH + "A3.png")

    img = read_image(INPUT_PATH + "B3.jpg")
    outputs = segmentation_mean_shift_for_B3(img, 0.05, 100)
    write_image(outputs[0], OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_0.05_overlay_only.png")
    write_image(outputs[1], OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_0.05_segmented_only.png")

    img = read_image(INPUT_PATH + "B3.jpg")
    outputs = segmentation_mean_shift_for_B3(img, 0.2, 50)
    write_image(outputs[0], OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_0.2_overlay_only.png")
    write_image(outputs[1], OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_0.2_segmented_only.png")

    img = read_image(INPUT_PATH + "B3.jpg")
    outputs = segmentation_mean_shift_for_B3(img, 0.3, 100)
    write_image(outputs[0], OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_0.3_overlay_only.png")
    write_image(outputs[1], OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_0.3_segmented_only.png")

    img = read_image(INPUT_PATH + "B2.jpg")
    outputs = segmentation_mean_shift_for_others(img, 0.05, 100)
    write_image(outputs[0], OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_0.01_overlay_only.png")
    write_image(outputs[1], OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_0.01_segmented_only.png")

    img = read_image(INPUT_PATH + "B2.jpg")
    outputs = segmentation_mean_shift_for_others(img, 0.2, 50)
    write_image(outputs[0], OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_0.1_overlay_only.png")
    write_image(outputs[1], OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_0.1_segmented_only.png")

    img = read_image(INPUT_PATH + "B2.jpg")
    outputs = segmentation_mean_shift_for_others(img, 0.3, 100)
    write_image(outputs[0], OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_0.2_overlay_only.png")
    write_image(outputs[1], OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_0.2_segmented_only.png")


    img = read_image(INPUT_PATH + "B3.jpg")
    outputs = segmentation_n_cut_for_B3(img, 50, 16)
    write_image(outputs[0], OUTPUT_PATH + "B2_algorithm_ncut_parameterset_(50,16)_segmented.png")
    write_image(outputs[1], OUTPUT_PATH + "B2_algorithm_ncut_parameterset_(50,16)_overlay.png")

    img = read_image(INPUT_PATH + "B3.jpg")
    outputs = segmentation_n_cut_for_B3(img, 30, 16)
    write_image(outputs[0], OUTPUT_PATH + "B2_algorithm_ncut_parameterset_(30,16)_segmented.png")
    write_image(outputs[1], OUTPUT_PATH + "B2_algorithm_ncut_parameterset_(30,16)_overlay.png")

    img = read_image(INPUT_PATH + "B3.jpg")
    outputs = segmentation_n_cut_for_B3(img, 30, 16)
    write_image(outputs[0], OUTPUT_PATH + "B2_algorithm_ncut_parameterset_(30,8)_segmented.png")
    write_image(outputs[1], OUTPUT_PATH + "B2_algorithm_ncut_parameterset_(30,8)_overlay.png")
