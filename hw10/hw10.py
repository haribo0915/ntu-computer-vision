import numpy as np
import cv2


def conv(x, y):
    assert x.shape == y.shape
    return np.sum(x*y)

def padding(img):
    padding_img = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=int)
    padding_img[1:-1, 1:-1] = img[:, :] #middle part
    padding_img[0, 1:-1] = img[0, :]    #top row
    padding_img[-1, 1:-1] = img[-1, :]  #bottom row
    padding_img[1:-1, 0] = img[:, 0]    #left column
    padding_img[1:-1, -1] = img[:, -1]  #right column
    padding_img[0, 0] = img[0, 0]       #top-left element
    padding_img[0, -1] = img[0, -1]     #top-right element
    padding_img[-1, 0] = img[-1, 0]     #bottom-left element
    padding_img[-1, -1] = img[-1, -1]   #bottom-right element

    return padding_img

def conv_img(img, kernel, threshold):
    k = kernel.shape[0]
    padding_img = padding(img)
    padding_times = ((k-1)//2)-1
    for i in range(padding_times):
        padding_img = padding(padding_img)

    new_img = np.zeros(img.shape, dtype=int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = conv(padding_img[i:i+k, j:j+k], kernel)
            if val >= threshold:
                new_img[i][j] = 1
            elif val <= -threshold:
                new_img[i][j] = -1
            else:
                new_img[i][j] = 0

    return new_img

def zero_crossing(img):
    padding_img = padding(img)
    new_img = np.zeros(img.shape, dtype=int)
    dr = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc = [-1, 0, 1, -1, 1, -1, 0,1]

    for i in range(1, padding_img.shape[0]-1):
        for j in range(1, padding_img.shape[1]-1):
            new_img[i-1][j-1] = 255
            if padding_img[i][j] == 1:
                for k in range(8):
                    if (padding_img[i+dr[k]][j+dc[k]] == -1):
                        new_img[i-1][j-1] = 0
                        break
    return new_img


def main():
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    k1 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    k2 = np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ]) / 3
    k3 = np.array([
            [2, -1, 2],
            [-1, -4, -1],
            [2, -1, 2]
        ]) / 3
    k4 = np.array([
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
        ])
    k5 = np.array([
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ])
    cv2.imwrite('laplacian_mask1.bmp', zero_crossing(conv_img(img, k1, 15)))
    cv2.imwrite('laplacian_mask2.bmp', zero_crossing(conv_img(img, k2, 15)))
    cv2.imwrite('min_var_laplacian.bmp', zero_crossing(conv_img(img, k3, 20)))
    cv2.imwrite('laplacian_of_gaussian.bmp', zero_crossing(conv_img(img, k4, 3000)))
    cv2.imwrite('difference_of_gaussian.bmp', zero_crossing(conv_img(img, k5, 1)))

if __name__ == '__main__':
    main()