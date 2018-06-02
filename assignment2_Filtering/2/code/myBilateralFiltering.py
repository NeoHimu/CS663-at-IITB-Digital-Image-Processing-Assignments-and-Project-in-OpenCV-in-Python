import scipy.io
import cv2
import numpy as np
import sys,os
import h5py
import math
from copy import deepcopy 

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])

arrays = {}
f = h5py.File(img_path)
for k, v in f.items():
    arrays[k] = np.array(v)

#print(arrays['imageOrig'])
img = np.array(arrays['imageOrig']).T

#Method for creating a floating point image pixels
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
img = im2double(img)
orignal_image_for_comprision = deepcopy(img)

row,col = img.shape
mean = 0
sigma = 0.05# noise : standard deviation is 5% of the intensity range and intensity range is [0-1]
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
		SD_intensity_diff = 4.0#3#2.962886#np.std(intensity_diff)
		#sum_I_SD += SD_intensity_diff
		SD_spatial_diff = 1.1#1#0.096616#np.std(spatial_diff)
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
		
print("RMSD (4,1.1) %f"%((sum((result - orignal_image_for_comprision).ravel()**2)/(row*col))**0.5))

cv2.imshow('Bifiltered Image (4,1.1)',result)
cv2.imshow('Noisy Image',unchanged_noisy_image)
cv2.waitKey(0)

'''
RMSD (4,1) 0.064519
RMSD (4,1.1) 0.066763
RMSD (4,0.9) 0.061862
RMSD (3.6,1) 0.064565
RMSD (4.4,1) 0.064579

'''
