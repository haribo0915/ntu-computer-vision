import sys
import numpy as np
import cv2

def binarize(img):
    r = img.shape[0]
    c = img.shape[1]
    for i in range(r):
        for j in range(c):
            if img[i][j] >= 128:
                img[i][j] = 255
            else:
                img[i][j] = 0
    cv2.imwrite('binarized.bmp', img)

def comp(img):
    r = img.shape[0]
    c = img.shape[1]
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if img[i][j] == 255:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255
    return new_img

def dilation(img, kernel):
    r = img.shape[0]
    c = img.shape[1]
    new_img = np.zeros(img.shape, dtype=int)
    for i in range(r):
        for j in range(c):
            if img[i][j] == 255:
                for d in kernel:
                    dr, dc = d
                    if (i+dr) >= 0 and (i+dr) < r and (j+dc) >= 0 and (j+dc) < c:
                        new_img[i+dr][j+dc] = 255
    return new_img

def erosion(img, kernel):
    r = img.shape[0]
    c = img.shape[1]
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            flag = 1   
            for d in kernel:
                dr, dc = d
                if (i+dr) < 0 or (i+dr) >= r or (j+dc) < 0 or (j+dc) >= c or img[i+dr][j+dc] == 0:
                    flag = 0
                    break
            if flag:
                new_img[i][j] = 255
    return new_img                

def close(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def open(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def hitAndMiss(img):
    J_kernel = [[0, -1], [0, 0], [1, 0]]
    K_kernel = [[-1, 0], [-1, 1], [0, 1]]
    img_comp = comp(img)
    r = img.shape[0]
    c = img.shape[1]
    new_img = np.zeros(img.shape, dtype=int)
    tmp1 = erosion(img, J_kernel)
    tmp2 = erosion(img_comp, K_kernel)
    for i in range(r):
        for j in range(c):
            if (tmp1[i][j] == 255 and tmp2[i][j] == 255):
                new_img[i][j] = 255
    return new_img


if __name__ == '__main__':
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    binarize(img)

    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]

    img = cv2.imread('binarized.bmp', cv2.IMREAD_GRAYSCALE)
    img_dil = dilation(img, kernel)
    cv2.imwrite('dilation.bmp', img_dil)
    img_ero = erosion(img, kernel)
    cv2.imwrite('erosion.bmp', img_ero)
    img_close = close(img, kernel)
    cv2.imwrite('close.bmp', img_close)
    img_open = open(img, kernel)
    cv2.imwrite('open.bmp', img_open)
    img_hm = hitAndMiss(img)
    cv2.imwrite('hit_and_miss.bmp', img_hm)