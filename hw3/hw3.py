import numpy as np
import matplotlib.pyplot as plt
import cv2

def	plotHist(img, name):
	if name == 'dark':
		k = 3
	else:
		k = 1
	r = img.shape[0]
	c = img.shape[1]
	x = np.zeros(256, dtype=int)
	for i in range(r):
		for j in range(c):
			intensity = img[i][j] // k
			x[intensity] += 1
	plt.bar(np.arange(256), x)
	plt.savefig("histogram_"+name+".png")
	plt.close()

def dark(img):
	r = img.shape[0]
	c = img.shape[1]
	for i in range(r):
		for j in range(c):
			img[i][j] /= 3
	cv2.imwrite('dark.bmp', img)

def equalization(img):
	r = img.shape[0]
	c = img.shape[1]
	x = np.zeros(256, dtype=int)
	prefix = np.zeros(256, dtype=int)
	s = np.zeros(256, dtype=int)
	for i in range(r):
		for j in range(c):
			x[img[i][j]] += 1
	prefix[0] = x[0]
	for i in range(1,len(x)):
		prefix[i] = x[i] + prefix[i-1]

	for i in range(r):
		for j in range(c):
			img[i][j] = 255*(prefix[img[i][j]] / (r*c))
	cv2.imwrite('equalization.bmp', img)	



if __name__ == '__main__':
	img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	plotHist(img, 'original')
	plotHist(img, 'dark')
	dark(img)
	img = cv2.imread('dark.bmp', cv2.IMREAD_GRAYSCALE)
	equalization(img)
	img = cv2.imread('equalization.bmp', cv2.IMREAD_GRAYSCALE)
	plotHist(img, 'equalization')
	