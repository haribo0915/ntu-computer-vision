import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from PIL import Image, ImageDraw

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

def	plotHist(img):
	r = img.shape[0]
	c = img.shape[1]
	x = np.zeros(256, dtype=int)
	for i in range(r):
		for j in range(c):
			x[img[i][j]] += 1
	plt.bar(np.arange(256), x)
	plt.savefig("histogram.png")
	plt.close()

def connectedComponent(img):
	r = img.shape[0]
	c = img.shape[1]
	#padding
	MAX = r*c+1
	padding_img = np.zeros((r+2, c+2))
	for i in range(c+2):
		padding_img[0][i] = MAX
		padding_img[r+1][i] = MAX
	for i in range(r+2):
		padding_img[i][0] = MAX
		padding_img[i][c+1] = MAX

	#initialize
	label = 1
	for i in range(r):
		for j in range(c):
			if img[i][j] == 255:
				padding_img[i+1][j+1] = label
				label += 1

	dr = [-1, 1, 0, 0]
	dc = [0, 0, -1, 1]
	changed = True
	while (changed == True):
		changed = False
		#top-down
		for i in range(1,r+1):
			for j in range(1,c+1):
				if padding_img[i][j] != 0:
					m = padding_img[i][j]
					for k in range(4):
						pr = i + dr[k]
						pc = j + dc[k]
						if padding_img[pr][pc] != 0:
							m = min(m, padding_img[pr][pc])
					if (m != padding_img[i][j]):
						padding_img[i][j] = m
						changed = True
		#bottom-up
		for i in range(r,0,-1):
			for j in range(c,0,-1):
				if padding_img[i][j] != 0:
					m = padding_img[i][j]
					for k in range(4):
						pr = i + dr[k]
						pc = j + dc[k]
						if padding_img[pr][pc] != 0:
							m = min(m, padding_img[pr][pc])
					if (m != padding_img[i][j]):
						padding_img[i][j] = m
						changed = True
	raw_dict = {}
	for i in range(1,r+1):
		for j in range(1,c+1):
			label = padding_img[i][j]
			if label != 0:
				if label in raw_dict:
					raw_dict[label].append((i-1,j-1))
				else:
					raw_dict[label] = [(i-1,j-1)]
	dict_ = {}
	cnt = 0
	for key in raw_dict:
		if len(raw_dict[key]) >= 500:
			dict_[cnt] = raw_dict[key].copy()
			cnt += 1
	return dict_

def draw_centroid(img, pos, color):
    x, y = pos
    width, height = 11, 11
    for r in range(-5, 5 + 1):
        img.putpixel((x + r, y), color)
        img.putpixel((x, y + r), color)	

def draw(dict_):
	img = Image.open('binarized.bmp')
	connect_img = Image.new("RGB", img.size)
	connect_img.paste(img)
	#print(connect_img.mode)
	for key in dict_:
		(r1,c1), (r2,c2) = dict_[key][0], dict_[key][0] 
		centroid_r, centroid_c = 0, 0
		for point in dict_[key]:
			if point[0] < r1:
				r1 = point[0]
			if point[0] > r2:
				r2 = point[0]
			if point[1] < c1:
				c1 = point[1]
			if point[1] > c2:
				c2 = point[1]
			centroid_r += point[0]
			centroid_c += point[1]
		centroid_r /= len(dict_[key])
		centroid_c /= len(dict_[key])
		draw = ImageDraw.Draw(connect_img)
		draw.rectangle([c1,r1,c2,r2], width = 3, outline="blue")
		draw_centroid(connect_img, (int(centroid_c), int(centroid_r)), (255,0,0))
	connect_img.save('connected.bmp')

if __name__ == '__main__':
	img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	plotHist(img)
	binarize(img)

	img = cv2.imread('binarized.bmp', cv2.IMREAD_GRAYSCALE)
	dict_ = connectedComponent(img)
	draw(dict_)
