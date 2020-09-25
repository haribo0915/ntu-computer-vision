import sys
import numpy as np
import cv2

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

def close(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def open(img, kernel):
    return dilation(erosion(img, kernel), kernel)

if __name__ == '__main__':
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]

    img_dil = dilation(img, kernel)
    cv2.imwrite('grayDilation.bmp', img_dil)
    img_ero = erosion(img, kernel)
    cv2.imwrite('grayErosion.bmp', img_ero)
    img_close = close(img, kernel)
    cv2.imwrite('grayClose.bmp', img_close)
    img_open = open(img, kernel)
    cv2.imwrite('grayOpen.bmp', img_open)