import numpy as np
import cv2

def binarize(img):
	new_img = np.zeros(img.shape, dtype=int)
	r = img.shape[0]
	c = img.shape[1]
	for i in range(r):
		for j in range(c):
			if img[i][j] >= 128:
				new_img[i][j] = 255
			else:
				new_img[i][j] = 0
	return new_img

def downSample(img):
	new_img = np.zeros((64,64), dtype=int)
	r = img.shape[0]
	c = img.shape[1]
	for i in range(0, r, 8):
		for j in range(0, c, 8):
			new_img[i//8][j//8] = img[i][j]
	return new_img

def Yokoi(img):
	new_img = np.zeros(img.shape, dtype=int)
	r = img.shape[0]
	c = img.shape[1]
	dr = [[0, -1, -1], [-1, -1, 0], [0, 1, 1], [1, 1, 0]]
	dc = [[1, 1, 0], [0, -1, -1], [-1, -1, 0], [0, 1, 1]]
	for i in range(r):
		for j in range(c):
			if (img[i][j] == 255):
				R, q = 0, 0
				for k in range(4):
					cnt = 0
					for l in range(3):
						pr, pc = i+dr[k][l], j+dc[k][l]
						if l == 0:
							if ((pr < 0) or (pr >= r) or (pc < 0) or (pc >= c) or (img[pr][pc] != 255)):
								break
							else:
								cnt = 1
						else:
							if ((pr >= 0 and pr < r and pc >= 0 and pc < c) and (img[pr][pc] == 255)):
								cnt += 1
					if cnt == 3:
						R += 1
					elif 0 < cnt and cnt < 3:
						q += 1
				if R == 4:
					new_img[i][j] = 5
				else:
					new_img[i][j] = q
			else:
				new_img[i][j] = 0
	return new_img

def Pair_relation(img):
	new_img = np.zeros(img.shape, dtype=int)
	r = img.shape[0]
	c = img.shape[1]
	dr = [0, -1, 0, 1]
	dc = [1, 0, -1, 0]
	for i in range(r):
		for j in range(c):
			if img[i][j] == 1:
				cnt = 0
				for k in range(4):
					pr, pc = i+dr[k], j+dc[k]
					if ((pr >= 0 and pr < r and pc >= 0 and pc < c) and (img[pr][pc] == 1)):
						cnt += 1
				if cnt >= 1:
					new_img[i][j] = 1
				else:
					cnt = 0
			else:
				new_img[i][j] = 0
	return new_img

def shrink(img, pair_relation_img):
	r = img.shape[0]
	c = img.shape[1]
	dr = [[0, -1, -1], [-1, -1, 0], [0, 1, 1], [1, 1, 0]]
	dc = [[1, 1, 0], [0, -1, -1], [-1, -1, 0], [0, 1, 1]]
	for i in range(r):
		for j in range(c):
			if (pair_relation_img[i][j] == 1):
				R, q = 0, 0
				for k in range(4):
					cnt = 0
					for l in range(3):
						pr, pc = i+dr[k][l], j+dc[k][l]
						if l == 0:
							if ((pr < 0) or (pr >= r) or (pc < 0) or (pc >= c) or (img[pr][pc] != 255)):
								break
							else:
								cnt = 1
						else:
							if ((pr >= 0 and pr < r and pc >= 0 and pc < c) and (img[pr][pc] == 255)):
								cnt += 1
					if cnt == 3:
						R += 1
					elif 0 < cnt and cnt < 3:
						q += 1
				if q == 1:
					img[i][j] = 0
	return img

if __name__ == '__main__':
	img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
	new_img = downSample(binarize(img))
	for i in range(7):
		yokoi_img = Yokoi(new_img)
		pair_relation_img = Pair_relation(yokoi_img)
		new_img = shrink(new_img, pair_relation_img)
	cv2.imwrite('thinning.bmp', new_img)
