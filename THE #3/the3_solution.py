# Onur Yaman , 2007961
# İlkim Oğul , 2237675
import os
from skimage import io, filters
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


INPUT_PATH = "./THE3_Images/"
OUTPUT_PATH = "./Outputs/"


def read_image(img_path, rgb=True):
    img = cv2.imread(img_path)
    if rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_image(img, output_path, rgb=False):
    plt.imshow(img)
    plt.savefig(output_path)

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def color_images(source_file, target_file, rescale=True):
    # read images as RGB
    source_rgb = Image.open(source_file).convert('RGB')
    target_rgb = Image.open(target_file).convert('RGB')
    # convert RGB to LAB
    source_lab = rgb2lab(np.uint8(source_rgb))
    target_lab = rgb2lab(np.uint8(target_rgb))
    # calculate mean and standard deviation of LAB images
    source_mu, source_sigma = _img_stats(source_lab)
    target_mu, target_sigma = _img_stats(target_lab)
    # ensure standard deviations to be non-zero to avoid divide-by-zero error
    source_sigma = np.where(source_sigma == 0,
                               np.ones_like(source_sigma, np.float32) * 1e-4,
                               source_sigma)
    target_sigma = np.where(target_sigma == 0,
                               np.ones_like(target_sigma, np.float32) * 1e-4,
                               target_sigma)
    # subtract mean of target from target
    target_lab -= target_mu
    # scale target using standard deviations
    target_lab *= (target_sigma / source_sigma)
    # add mean of source to target
    target_lab += source_mu
    # convert LAB to RGB
    result_rgb = lab2rgb(target_lab)
    if rescale:
        result_rgb = np.uint8(_rescale(result_rgb))
    else:
        result_rgb = np.uint8(result_rgb)
    result_rgb = Image.fromarray(result_rgb)
    return result_rgb

def rgb2lab(array):
    assert len(array.shape) == 3 and array.shape[2] == 3, \
        'Input array needs to be a RGB image.'
    # initialize transformation matrices
    T1_RGB2LMS = np.float32([[0.3811, 0.5783, 0.0402],
                                [0.1967, 0.7244, 0.0782],
                                [0.0241, 0.1288, 0.8444]])
    T2_LMS2LAB = np.float32([[1., 1., 1.],
                                [1., 1., -2.],
                                [1., -1., 0.]])
    T3_LMS2LAB = np.float32([[1. / np.sqrt(3.), 0., 0.],
                                [0., 1. / np.sqrt(6.), 0.],
                                [0., 0., 1. / np.sqrt(2.)]])
    RGB = np.float32(array).reshape(-1, 3).T
    LMS = np.matmul(T1_RGB2LMS, RGB)
    LMS = np.where(LMS == 0, np.ones_like(LMS, np.float32) * 1e-4, LMS)
    LMS = np.log10(LMS)
    LAB = np.matmul(T3_LMS2LAB, np.matmul(T2_LMS2LAB, LMS))
    LAB = LAB.T.reshape(array.shape[0], array.shape[1], 3)
    return LAB

def lab2rgb(array):
    assert len(array.shape) == 3 and array.shape[2] == 3, \
        'Input array needs to be a LAB image.'
    T1_LAB2LMS = np.float32([[np.sqrt(3) / 3., 0., 0.],
                                [0., np.sqrt(6) / 6., 0.],
                                [0., 0., np.sqrt(2) / 2.]])
    T2_LAB2LMS = np.float32([[1., 1., 1.],
                                [1., 1., -1.],
                                [1., -2., 0.]])
    T3_LMS2RGB = np.float32([[4.4679, -3.5873, 0.1193],
                                [-1.2186, 2.3809, -0.1624],
                                [0.0497, -0.2439, 1.2045]])
    LAB = np.float32(array).reshape(-1, 3).T
    LMS = np.matmul(T2_LAB2LMS, np.matmul(T1_LAB2LMS, LAB))
    LMS = 10. ** LMS
    RGB = np.matmul(T3_LMS2RGB, LMS)
    RGB = RGB.T.reshape(array.shape[0], array.shape[1], 3)
    return RGB

def _img_stats(img):
    img = np.float32(img).reshape(-1, 3).T
    mu = np.mean(img, axis=1, keepdims=False)
    sigma = np.std(img, axis=1, keepdims=False)
    return (mu, sigma)

def _rescale(img):
    img = np.float32(img)
    img = 255. * (img - img.min()) / (img.max() - img.min())
    return img

def detect_edges(img):
    sobel_img = filters.sobel(img)
    return sobel_img

def rgb2hsi(img):
    with np.errstate(divide='ignore', invalid='ignore'):
        bgr = np.float32(img)/255
        blue = bgr[:, :, 0]
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)
            return saturation
        def calc_hue(red, blue, green):
            hue = np.copy(red)
            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])
                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]
            return hue
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return hsi


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    img = read_image(INPUT_PATH + "1_source.png")
    output = detect_faces(img)
    write_image(output, OUTPUT_PATH + "1_faces")

    img = read_image(INPUT_PATH + "2_source.png")
    output = detect_faces(img)
    write_image(output, OUTPUT_PATH + "2_faces")

    img = read_image(INPUT_PATH + "3_source.png")
    output = detect_faces(img)
    write_image(output, OUTPUT_PATH + "3_faces")

    output1 = color_images(INPUT_PATH + "1_source.png", INPUT_PATH + "1.png")
    write_image(output1, OUTPUT_PATH + "1_colored_RGB.png")
    edges = detect_edges(np.array(output1))
    write_image(edges, OUTPUT_PATH + "1_colored_RGB_edges.png")

    output2 = color_images(INPUT_PATH + "2_source.png", INPUT_PATH + "2.png")
    write_image(output2, OUTPUT_PATH + "2_colored_RGB.png")
    edges = detect_edges(np.array(output2))
    write_image(edges, OUTPUT_PATH + "2_colored_RGB_edges.png")

    output3 = color_images(INPUT_PATH + "3_source.png", INPUT_PATH + "3.png")
    write_image(output3, OUTPUT_PATH + "3_colored_RGB.png")
    edges = detect_edges(np.array(output3))
    write_image(edges, OUTPUT_PATH + "3_colored_RGB_edges.png")

    output4 = color_images(INPUT_PATH + "4_source.png", INPUT_PATH + "4.png")
    write_image(output4, OUTPUT_PATH + "4_colored_RGB.png")
    edges = detect_edges(np.array(output4))
    write_image(edges, OUTPUT_PATH + "4_colored_RGB_edges.png")

    output1_hsi = rgb2hsi(output1)
    write_image(output1_hsi, OUTPUT_PATH + "1_colored_HSI.png")
    edges = detect_edges(np.array(output1_hsi))
    write_image(edges, OUTPUT_PATH + "1_colored_HSI_edges.png")

    output2_hsi = rgb2hsi(output2)
    write_image(output2_hsi, OUTPUT_PATH + "2_colored_HSI.png")
    edges = detect_edges(np.array(output2_hsi))
    write_image(edges, OUTPUT_PATH + "2_colored_HSI_edges.png")

    output3_hsi = rgb2hsi(output3)
    write_image(output3_hsi, OUTPUT_PATH + "3_colored_HSI.png")
    edges = detect_edges(np.array(output3_hsi))
    write_image(edges, OUTPUT_PATH + "3_colored_HSI_edges.png")

    output4_hsi = rgb2hsi(output4)
    write_image(output4_hsi, OUTPUT_PATH + "4_colored_HSI.png")
    edges = detect_edges(np.array(output4_hsi))
    write_image(edges, OUTPUT_PATH + "4_colored_HSI_edges.png")

    img = read_image(INPUT_PATH + "1_source.png")
    output = detect_edges(img)
    write_image(output, OUTPUT_PATH + "1_edges.png")

    img = read_image(INPUT_PATH + "2_source.png")
    output = detect_edges(img)
    write_image(output, OUTPUT_PATH + "2_edges.png")

    img = read_image(INPUT_PATH + "3_source.png")
    output = detect_edges(img)
    write_image(output, OUTPUT_PATH + "3_edges.png")

    img = read_image(INPUT_PATH + "4_source.png")
    output = detect_edges(img)
    write_image(output, OUTPUT_PATH + "4_edges.png")