#ilkim oğul, 2237675
#onur yaman, 2007961

import os
from skimage import io, transform, data, color, exposure
import math
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize


INPUT_PATH = "./THE1_Images/"
OUTPUT_PATH = "./Outputs/"

def read_image(img_path, rgb = True):
    img = io.imread(img_path)
    return img


def write_image(img, output_path, rgb = True):
    if rgb == True:
        plt.imshow(img)
        plt.savefig(output_path)
    elif rgb== False:
        plt.imshow(img,cmap='gray')
        plt.savefig(output_path)


def rotate_image(img,  degree = 0, interpolation_type = "linear"):
    tf_img=transform.rotate(img,degree,resize=True)
    if interpolation_type == "linear":
        i_img = rescale(tf_img, 1, anti_aliasing=False, channel_axis=True, order = 1)
        return i_img
    elif interpolation_type == "cubic":
        i_img = rescale(tf_img, 1, anti_aliasing=False, channel_axis=True, order = 3)
        return i_img


def extract_save_histogram(img, path):
    plt.figure(figsize=(12, 10))
    plt.hist(img.ravel(),bins=20)
    plt.savefig(path)

def histogram_equalization(img):
    img_hist_eq=exposure.equalize_hist(img)
    return img_hist_eq

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #PART1

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a1_45_cubic.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a1_90_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "cubic")
    write_image(output, OUTPUT_PATH + "a1_90_cubic.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a2_45_linear.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a2_45_cubic.png")


    img = read_image(INPUT_PATH + "b1.png", rgb = False)
    extract_save_histogram(img, OUTPUT_PATH + "original_histogram.png")
    equalized = histogram_equalization(img)
    extract_save_histogram(equalized, OUTPUT_PATH + "equalized_histogram.png")
    write_image(equalized, OUTPUT_PATH + "enhanced_image.png", rgb = False)
