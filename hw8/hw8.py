import os
import copy
import numpy as np
import cv2
from math import sqrt, log

def gaussianNoise(img):
    gaussian_10 = img + 10 * np.random.normal(0, 1, img.shape)
    cv2.imwrite('P1/gaussian_10.bmp', gaussian_10)
    gaussian_30 = img + 30 * np.random.normal(0, 1, img.shape)
    cv2.imwrite('P1/gaussian_30.bmp', gaussian_30)

def saltPepperNoise(img):
    (r, c) = img.shape
    prob = np.random.uniform(0, 1, img.shape)
    new_img1 = np.zeros(img.shape, dtype=int)
    new_img2 = np.zeros(img.shape, dtype=int)
    threshold = 0.1
    for i in range(r):
        for j in range(c):
            new_img1[i][j] = img[i][j]
            new_img2[i][j] = img[i][j]
    for i in range(r):
        for j in range(c):
            if prob[i][j] < threshold:
                new_img1[i][j] = 0
            elif prob[i][j] > (1-threshold):
                new_img1[i][j] = 255
    cv2.imwrite('P2/sp_0.1.bmp', new_img1)

    threshold = 0.05
    for i in range(r):
        for j in range(c):
            if prob[i][j] < threshold:
                new_img2[i][j] = 0
            elif prob[i][j] > (1-threshold):
                new_img2[i][j] = 255
    cv2.imwrite('P2/sp_0.05.bmp', new_img2)            

def box(img, size):
    (r, c) = img.shape
    new_img = np.zeros(shape=(r-size+1, c-size+1),dtype=np.float32)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i][j] = np.mean(img[i:i+size, j:j+size])
    return new_img

def median(img, size):
    (r, c) = img.shape
    new_img = np.zeros(shape=(r-size+1, c-size+1),dtype=np.float32)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i][j] = np.median(img[i:i+size, j:j+size])
    return new_img

def dilation(img, kernel):
    r = img.shape[0]
    c = img.shape[1]
    new_img = np.zeros(img.shape, dtype=int)
    for i in range(r):
        for j in range(c):
            localMax = 0
            for d in kernel:
                dr, dc = d
                if (i+dr) >= 0 and (i+dr) < r and (j+dc) >= 0 and (j+dc) < c:
                    localMax = max(img[i+dr][j+dc], localMax)
            new_img[i][j] = localMax
    return new_img

def erosion(img, kernel):
    r = img.shape[0]
    c = img.shape[1]
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            localMin = 255
            for d in kernel:
                dr, dc = d
                if (i+dr) >= 0 and (i+dr) < r and (j+dc) >= 0 and (j+dc) < c:
                    localMin = min(img[i+dr][j+dc], localMin)
            new_img[i][j] = localMin
    return new_img                

def close_(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def open_(img, kernel):
    return dilation(erosion(img, kernel), kernel)    

def SNR(img, noise_img):
    img = img / 255
    noise_img = noise_img / 255
    if noise_img.shape[0] < img.shape[0]:
        padding = (img.shape[0]-noise_img.shape[0])//2
        img = img[padding:img.shape[0]-padding, padding:img.shape[0]-padding]

    vs = np.var(img)
    tmp = noise_img-img
    vn = np.var(tmp)
    return 20*log(sqrt(vs)/sqrt(vn), 10)

if __name__ == '__main__':
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]

    for i in range(1, 6):
        if not os.path.exists('P'+str(i)):
            os.makedirs('P'+str(i))

    gaussianNoise(img)
    saltPepperNoise(img)

    gaussian_10 = cv2.imread('P1/gaussian_10.bmp', cv2.IMREAD_GRAYSCALE)
    gaussian_30 = cv2.imread('P1/gaussian_30.bmp', cv2.IMREAD_GRAYSCALE)
    sp_01 = cv2.imread('P2/sp_0.1.bmp', cv2.IMREAD_GRAYSCALE)
    sp_005 = cv2.imread('P2/sp_0.05.bmp', cv2.IMREAD_GRAYSCALE)

    cv2.imwrite('P3/gaussian_10_box_3.bmp', box(gaussian_10, 3))
    cv2.imwrite('P3/gaussian_30_box_3.bmp', box(gaussian_30, 3))
    cv2.imwrite('P3/sp_0.05_box_3.bmp', box(sp_005, 3))
    cv2.imwrite('P3/sp_0.1_box_3.bmp', box(sp_01, 3))
    cv2.imwrite('P3/gaussian_10_box_5.bmp', box(gaussian_10, 5))
    cv2.imwrite('P3/gaussian_30_box_5.bmp', box(gaussian_30, 5))
    cv2.imwrite('P3/sp_0.05_box_5.bmp', box(sp_005, 5))
    cv2.imwrite('P3/sp_0.1_box_5.bmp', box(sp_01, 5))

    cv2.imwrite('P4/gaussian_10_median_3.bmp', median(gaussian_10, 3))
    cv2.imwrite('P4/gaussian_30_median_3.bmp', median(gaussian_30, 3))
    cv2.imwrite('P4/sp_0.05_median_3.bmp', median(sp_005, 3))
    cv2.imwrite('P4/sp_0.1_median_3.bmp', median(sp_01, 3))
    cv2.imwrite('P4/gaussian_10_median_5.bmp', median(gaussian_10, 5))
    cv2.imwrite('P4/gaussian_30_median_5.bmp', median(gaussian_30, 5))
    cv2.imwrite('P4/sp_0.05_median_5.bmp', median(sp_005, 5))
    cv2.imwrite('P4/sp_0.1_median_5.bmp', median(sp_01, 5))

    cv2.imwrite('P5/gaussian_10_close_open.bmp', open_(close_(gaussian_10, kernel), kernel))
    cv2.imwrite('P5/gaussian_30_close_open.bmp', open_(close_(gaussian_30, kernel), kernel))
    cv2.imwrite('P5/sp_0.05_close_open.bmp', open_(close_(sp_005, kernel), kernel))
    cv2.imwrite('P5/sp_0.1_close_open.bmp', open_(close_(sp_01, kernel), kernel))

    cv2.imwrite('P5/gaussian_10_open_close.bmp', close_(open_(gaussian_10, kernel), kernel))
    cv2.imwrite('P5/gaussian_30_open_close.bmp', close_(open_(gaussian_30, kernel), kernel))
    cv2.imwrite('P5/sp_0.05_open_close.bmp', close_(open_(sp_005, kernel), kernel))
    cv2.imwrite('P5/sp_0.1_open_close.bmp', close_(open_(sp_01, kernel), kernel))

    with open("SNR.txt", "w") as file:
        file.write("gaussianNoise_10_SNR: " + str(SNR(img, gaussian_10)) + '\n')
        file.write("gaussianNoise_30_SNR: " + str(SNR(img, gaussian_30))+ '\n')
        file.write("saltAndPepper_0_10_SNR: " + str(SNR(img, sp_01)) + '\n')
        file.write("saltAndPepper_0_05_SNR: " + str(SNR(img, sp_005)) + '\n')

        file.write("gaussianNoise_10_box_3x3_SNR: " + str(SNR(img, box(gaussian_10, 3))) + '\n')
        file.write("gaussianNoise_30_box_3x3_SNR: " + str(SNR(img, box(gaussian_30, 3))) + '\n')
        file.write("saltAndPepper_0_10_box_3x3_SNR: " + str(SNR(img, box(sp_01, 3))) + '\n')
        file.write("saltAndPepper_0_05_box_3x3_SNR: " + str(SNR(img, box(sp_005, 3))) + '\n')
        file.write("gaussianNoise_10_box_5x5_SNR: " + str(SNR(img, box(gaussian_10, 5))) + '\n')
        file.write("gaussianNoise_30_box_5x5_SNR: " + str(SNR(img, box(gaussian_30, 5))) + '\n')
        file.write("saltAndPepper_0_10_box_5x5_SNR: " + str(SNR(img, box(sp_01, 5))) + '\n')
        file.write("saltAndPepper_0_05_box_5x5_SNR: " + str(SNR(img, box(sp_005, 5))) + '\n')

        file.write("gaussianNoise_10_median_3x3_SNR: " + str(SNR(img, median(gaussian_10, 3))) + '\n')
        file.write("gaussianNoise_30_median_3x3_SNR: " + str(SNR(img, median(gaussian_30, 3))) + '\n')
        file.write("saltAndPepper_0_10_median_3x3_SNR: " + str(SNR(img, median(sp_01, 3))) + '\n')
        file.write("saltAndPepper_0_05_median_3x3_SNR: " + str(SNR(img, median(sp_005, 3))) + '\n')
        file.write("gaussianNoise_10_median_5x5_SNR: " + str(SNR(img, median(gaussian_10, 5))) + '\n')
        file.write("gaussianNoise_30_median_5x5_SNR: " + str(SNR(img, median(gaussian_30, 5))) + '\n')
        file.write("saltAndPepper_0_10_median_5x5_SNR: " + str(SNR(img, median(sp_01, 5))) + '\n')
        file.write("saltAndPepper_0_05_median_5x5_SNR: " + str(SNR(img, median(sp_005, 5))) + '\n')
        
        file.write("gaussianNoise_10_openingThenClosing_SNR: " + str(SNR(img, close_(open_(gaussian_10, kernel), kernel))) + '\n')
        file.write("gaussianNoise_30_openingThenClosing_SNR: " + str(SNR(img, close_(open_(gaussian_30, kernel), kernel))) + '\n')
        file.write("saltAndPepper_0_10_openingThenClosing_SNR: " + str(SNR(img, close_(open_(sp_01, kernel), kernel))) + '\n')
        file.write("saltAndPepper_0_05_openingThenClosing_SNR: " + str(SNR(img, close_(open_(sp_005, kernel), kernel))) + '\n')

        file.write("gaussianNoise_10_closingThenOpening_SNR: " + str(SNR(img, open_(close_(gaussian_10, kernel), kernel))) + '\n')
        file.write("gaussianNoise_30_closingThenOpening_SNR: " + str(SNR(img, open_(close_(gaussian_30, kernel), kernel))) + '\n')
        file.write("saltAndPepper_0_10_closingThenOpening_SNR: " + str(SNR(img, open_(close_(sp_01, kernel), kernel))) + '\n')
        file.write("saltAndPepper_0_05_closingThenOpening_SNR: " + str(SNR(img, open_(close_(sp_005, kernel), kernel))) + '\n')