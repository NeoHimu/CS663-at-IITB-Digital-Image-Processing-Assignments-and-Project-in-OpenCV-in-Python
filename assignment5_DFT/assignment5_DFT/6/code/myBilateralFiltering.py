import scipy.io
import cv2
import numpy as np
import sys,os
import h5py
import math
from copy import deepcopy 

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/barbara256-part.png')
img = cv2.imread(img_path, 0) # 0 : Gray image
#Method for creating a floating point image pixels
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
#img = im2double(img)
orignal_image_for_comprision = deepcopy(img)

row,col = img.shape
mean = 0
sigma = 20.0#/255.0
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
#print(gauss)
noisy = img + gauss
unchanged_noisy_image = deepcopy(noisy)
window_size = 10
result = np.zeros(noisy.shape)
count = 0
#print("%d %d"%(row,col))

def gaussian(x, sigma):
	return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
sum_I_SD = 0.0
sum_S_SD = 0.0
for idx1 in range(row):
	for idx2 in range(col):
		img_temp = noisy[(0 if idx1-window_size<0 else (idx1-window_size)):row-1 if (idx1+window_size+1)>row-1 else \
		(idx1+window_size+1),0 if (idx2-window_size)<0 else (idx2-window_size): col-1 if (idx2+window_size+1)>col-1 \
		else (idx2+window_size+1)]
		
		intensity_diff = abs(img_temp - np.ones((img_temp.shape))*noisy[idx1][idx2])
		
		spatial_diff = np.zeros((img_temp.shape))
		row_temp,col_temp = spatial_diff.shape
		for idx3 in range(row_temp):
			for idx4 in range(col_temp):
				spatial_diff[idx3][idx4] = ((idx3 - row_temp/2)**2+(idx4 - col_temp/2)**2)**0.5 #calculating the euclidean dist
		#Standard deviations
		SD_intensity_diff = 60
		#sum_I_SD += SD_intensity_diff
		SD_spatial_diff = 40
		#sum_S_SD += SD_spatial_diff
		Wp = 0.0
		Ip = 0.0
		
		for idx3 in range(row_temp):
			for idx4 in range(col_temp):
				gauss_i = gaussian(intensity_diff[idx3][idx4],SD_intensity_diff)
				gauss_s = gaussian(spatial_diff[idx3][idx4],SD_spatial_diff)
				
				w = gauss_i*gauss_s
				filtered_intensity = w*img_temp[idx3][idx4]
				Wp += w
				Ip += filtered_intensity
		
		count += 1
		if count%1000 == 0:
			print(count)
			
		result[idx1][idx2] = float(Ip)/Wp	
		
print("MSD for Result and original %f"%((sum((result - orignal_image_for_comprision).ravel()**2)/(row*col))))
#print("MSD for noisy and original %f"%((sum((unchanged_noisy_image - orignal_image_for_comprision).ravel()**2)/(row*col))))
cv2.imshow('Bifiltered Image',im2double(result))
cv2.imshow('Noisy Image',im2double(unchanged_noisy_image))
cv2.waitKey(0)
