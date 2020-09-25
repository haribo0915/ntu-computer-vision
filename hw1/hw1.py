import numpy as np
import cv2
import sys

def upsideDown(img):
	r = img.shape[0]
	c = img.shape[1]
	new_img = []
	for i in range(r-1,-1,-1):
		tmp = []
		for j in range(c):
			tmp.append(img[i][j])
		new_img.append(tmp)
	new_img = np.array(new_img)
	cv2.imwrite('upsideDown.bmp', new_img)

def rightSideLeft(img):
	r = img.shape[0]
	c = img.shape[1]
	new_img = []
	for i in range(r):
		tmp = []
		for j in range(c):
			tmp.append(img[i][j])
		tmp.reverse()
		new_img.append(tmp)
	new_img = np.array(new_img)
	cv2.imwrite('rightSideLeft.bmp', new_img)

def diagonallyMirrored(img):
	r = img.shape[0]
	c = img.shape[1]
	for i in range(r):
		for j in range(i,c):
			tmp = np.array(img[i][j])
			img[i][j] = img[j][i]
			img[j][i] = tmp  
	cv2.imwrite('diagonallyMirrored.bmp', img)	

if __name__ == '__main__':
	img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
	upsideDown(img)
	rightSideLeft(img)
	diagonallyMirrored(img)
