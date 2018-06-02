import cv2
import numpy as np
import math

#img = cv2.imread('barbara.png')
#res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
#res = cv2.GaussianBlur(res, (5,5), 0.66)

def gaussian(x, sigma):
	return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


spatial_diff = np.zeros((25,25))
row_temp,col_temp = spatial_diff.shape
for idx3 in range(row_temp):
	for idx4 in range(col_temp):
		spatial_diff[idx3][idx4] = ((idx3 - row_temp/2)**2+(idx4 - col_temp/2)**2)**0.5 #calculating the euclidean dist
SD = 1.0
gauss_s = np.zeros((row_temp,col_temp))
for idx3 in range(0,row_temp):
	for idx4 in range(0,col_temp):
		gauss_s[idx3][idx4] = gaussian(spatial_diff[idx3][idx4],SD)
print(gauss_s[0][0])
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
gauss_s = im2double(gauss_s)
print(gauss_s[0][0])

cv2.imshow("Weight Filter Image",gauss_s)
cv2.waitKey(0)
