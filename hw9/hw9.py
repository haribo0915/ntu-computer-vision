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


def Roberts(img, threshold):
    padding_img = np.zeros((img.shape[0]+1, img.shape[1]+1), dtype=int)
    padding_img[:-1, :-1] = img[:, :] #middle part
    padding_img[:-1, -1] = img[:, -1] #right column
    padding_img[-1, :-1] = img[-1, :] #bottom row
    padding_img[-1, -1] = img[-1, -1] #bottom-right element

    new_img = np.zeros(img.shape, dtype=int)

    k1 = np.array([
        [1, 0],
        [0, -1]
    ])
    k2 = np.array([
        [0, 1],
        [-1, 0]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r1 = conv(padding_img[i:i+2, j:j+2], k1)
            r2 = conv(padding_img[i:i+2, j:j+2], k2)
            if np.sqrt(r1**2 + r2**2) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255
    
    return new_img

def Prewitt(img, threshold):
    padding_img = padding(img)
    new_img = np.zeros(img.shape, dtype=int)

    k1 = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    k2 = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p1 = conv(padding_img[i:i+3, j:j+3], k1)
            p2 = conv(padding_img[i:i+3, j:j+3], k2)
            if np.sqrt(p1**2 + p2**2) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img

def Sobel(img, threshold):
    padding_img = padding(img)
    new_img = np.zeros(img.shape, dtype=int)

    k1 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    k2 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s1 = conv(padding_img[i:i+3, j:j+3], k1)
            s2 = conv(padding_img[i:i+3, j:j+3], k2)
            if np.sqrt(s1**2 + s2**2) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img

def Frei(img, threshold):
    padding_img = padding(img)
    new_img = np.zeros(img.shape, dtype=int)

    k1 = np.array([
        [-1, -np.sqrt(2), -1],
        [0, 0, 0],
        [1, np.sqrt(2), 1]
    ])
    k2 = np.array([
        [-1, 0, 1],
        [-np.sqrt(2), 0, np.sqrt(2)],
        [-1, 0, 1]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            f1 = conv(padding_img[i:i+3, j:j+3], k1)
            f2 = conv(padding_img[i:i+3, j:j+3], k2)
            if np.sqrt(f1**2 + f2**2) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
    

def Kirsch(img, threshold):
    padding_img = padding(img)
    new_img = np.zeros(img.shape, dtype=int)

    k0 = np.array([
        [-3, -3, 5],
        [-3, 0, 5],
        [-3, -3, 5]
    ])
    k1 = np.array([
        [-3, 5, 5],
        [-3, 0, 5],
        [-3, -3, -3]
    ])
    k2 = np.array([
        [5, 5, 5],
        [-3, 0, -3],
        [-3, -3, -3]
    ])
    k3 = np.array([
        [5, 5, -3],
        [5, 0, -3],
        [-3, -3, -3]
    ])
    k4 = np.array([
        [5, -3, -3],
        [5, 0, -3],
        [5, -3, -3]
    ])
    k5 = np.array([
        [-3, -3, -3],
        [5, 0, -3],
        [5, 5, -3]
    ])
    k6 = np.array([
        [-3, -3, -3],
        [-3, 0, -3],
        [5, 5, 5]
    ])
    k7 = np.array([
        [-3, -3, -3],
        [-3, 0, 5],
        [-3, 5, 5]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c0 = conv(padding_img[i:i+3, j:j+3], k0)
            c1 = conv(padding_img[i:i+3, j:j+3], k1)
            c2 = conv(padding_img[i:i+3, j:j+3], k2)
            c3 = conv(padding_img[i:i+3, j:j+3], k3)
            c4 = conv(padding_img[i:i+3, j:j+3], k4)
            c5 = conv(padding_img[i:i+3, j:j+3], k5)
            c6 = conv(padding_img[i:i+3, j:j+3], k6)
            c7 = conv(padding_img[i:i+3, j:j+3], k7)
            if max(c0,c1,c2,c3,c4,c5,c6,c7) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img

def Robinson(img, threshold):
    padding_img = padding(img)
    new_img = np.zeros(img.shape, dtype=int)

    k0 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    k1 = np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]
    ])
    k2 = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    k3 = np.array([
        [2, 1, 0],
        [1, 0, -1],
        [0, -1, -2]
    ])
    k4 = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    k5 = np.array([
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]
    ])
    k6 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    k7 = np.array([
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c0 = conv(padding_img[i:i+3, j:j+3], k0)
            c1 = conv(padding_img[i:i+3, j:j+3], k1)
            c2 = conv(padding_img[i:i+3, j:j+3], k2)
            c3 = conv(padding_img[i:i+3, j:j+3], k3)
            c4 = conv(padding_img[i:i+3, j:j+3], k4)
            c5 = conv(padding_img[i:i+3, j:j+3], k5)
            c6 = conv(padding_img[i:i+3, j:j+3], k6)
            c7 = conv(padding_img[i:i+3, j:j+3], k7)
            if max(c0,c1,c2,c3,c4,c5,c6,c7) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img

def Nevatia(img, threshold):
    padding_img = padding(padding(img))
    new_img = np.zeros(img.shape, dtype=int)

    k0 = np.array([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [0, 0, 0, 0, 0],
        [-100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100],
    ])
    k1 = np.array([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 78, -32],
        [100, 92, 0, -92, -100],
        [32, -78, -100, -100, -100],
        [-100, -100, -100, -100, -100]
    ])
    k2 = np.array([
        [100, 100, 100, 32, -100],
        [100, 100, 92, -78, -100],
        [100, 100, 0, -100, -100],
        [100, 78, -92, -100, -100],
        [100, -32, -100, -100, -100]
    ])
    k3 = np.array([
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100]
    ])
    k4 = np.array([
        [-100, 32, 100, 100, 100],
        [-100, -78, 92, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, -92, 78, 100],
        [-100, -100, -100, -32, 100]
    ])
    k5 = np.array([
        [100, 100, 100, 100, 100],
        [-32, 78, 100, 100, 100],
        [-100, -92, 0, 92, 100],
        [-100, -100, -100, -78, 32],
        [-100, -100, -100, -100, -100]
    ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c0 = conv(padding_img[i:i+5, j:j+5], k0)
            c1 = conv(padding_img[i:i+5, j:j+5], k1)
            c2 = conv(padding_img[i:i+5, j:j+5], k2)
            c3 = conv(padding_img[i:i+5, j:j+5], k3)
            c4 = conv(padding_img[i:i+5, j:j+5], k4)
            c5 = conv(padding_img[i:i+5, j:j+5], k5)
            if max(c0,c1,c2,c3,c4,c5) >= threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img

def main():
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('Roberts.bmp', Roberts(img, 12))
    cv2.imwrite('Prewitt.bmp', Prewitt(img, 24))
    cv2.imwrite('Sobel.bmp', Sobel(img, 38))
    cv2.imwrite('Frei.bmp', Frei(img, 30))
    cv2.imwrite('Kirsch.bmp', Kirsch(img, 135))
    cv2.imwrite('Robinson.bmp', Robinson(img, 43))
    cv2.imwrite('Nevatia.bmp', Nevatia(img, 12500))

if __name__ == '__main__':
    main()