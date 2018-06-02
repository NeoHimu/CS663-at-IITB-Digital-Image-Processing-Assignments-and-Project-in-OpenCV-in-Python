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
orignal_image = deepcopy(img)
#Resizing and blurring
#img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
#img = cv2.GaussianBlur(img, (5,5), 0.66)


row,col = img.shape
mean = 0
sigma = 0.05# noise : standard deviation is 5% of the intensity range and intensity range is [0-1]
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
#print(gauss)
noisy = img + gauss
noisy_image = noisy
unchanged_noisy_image = deepcopy(noisy)
window_size = 14 # Actually window size is 2*14+1-4 = 25X25
patch_size = 4 # Actually patch size is 2*4+1 = 9X9
result = deepcopy(img)
count = 0
#print("%d %d"%(row,col))

def gaussian(x, sigma):
	return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
	
SD = 5.0 #Standard Deviation


# Gaussian Weight Filter : Isotropic filter
def gaussian(x, sigma):
	return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


spatial_diff = np.zeros((2*window_size+1,2*window_size+1))
row_temp,col_temp = spatial_diff.shape
for idx3 in range(row_temp):
	for idx4 in range(col_temp):
		spatial_diff[idx3][idx4] = ((idx3 - row_temp/2)**2+(idx4 - col_temp/2)**2)**0.5 #calculating the euclidean dist
SD_spatial_filter = 1.0
gauss_s = np.zeros((row_temp,col_temp))
for idx3 in range(0,row_temp):
	for idx4 in range(0,col_temp):
		gauss_s[idx3][idx4] = gaussian(spatial_diff[idx3][idx4],SD_spatial_filter)

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
#Normalize the isotropic filter
gauss_s = im2double(gauss_s)


for idx1 in range(window_size,row-window_size-1):
	for idx2 in range(window_size,col-window_size-1):
		#Selecting the window
		img_temp = noisy[(0 if idx1-window_size<0 else (idx1-window_size)):row-1 if (idx1+window_size+1)>row-1 else \
		(idx1+window_size+1),0 if (idx2-window_size)<0 else (idx2-window_size): col-1 if (idx2+window_size+1)>col-1 \
		else (idx2+window_size+1)]
		
		w_row,w_col = img_temp.shape
		center_r = w_row/2
		center_c = w_col/2
		#print(img_temp.shape)
		#choosing the center patch p
		p = img_temp[(center_r-patch_size):(center_r+patch_size+1),(center_c-patch_size):(center_c+patch_size+1)]
		#find other patches in the window
		denominator = 1.0
		#print(p.shape)
		final_value_of_p_intensity=0
		for idx3 in range(patch_size,w_row-patch_size-1):
			for idx4 in range(patch_size,w_col-patch_size-1):
				#Selecting a patch q
				q = img_temp[(idx3-patch_size):(idx3+patch_size+1),(idx4-patch_size):(idx4+patch_size+1)]
				numerator = math.exp(-sum((p.ravel()-q.ravel())**2)/SD**2)
				#print("%f %f"%(numerator,denominator))
				denominator += numerator*gauss_s[idx3][idx4]
				final_value_of_p_intensity += numerator*img_temp[idx3][idx4]*gauss_s[idx3][idx4]
				#This statement can be brought outside these two loops
				result[idx1][idx2] = float(final_value_of_p_intensity)/denominator
	
		count += 1
		if count%1000==0:
			print(count)
			

result = result[window_size:(row-window_size-1),window_size:(col-window_size-1)]
orignal_image = orignal_image[window_size:(row-window_size-1),window_size:(col-window_size-1)]
unchanged_noisy_image = unchanged_noisy_image[window_size:(row-window_size-1),window_size:(col-window_size-1)]

print("RMSD %f"%((sum((result - orignal_image).ravel()**2)/(row*col))**0.5))

cv2.imshow('Patch Based Filtered Image with SD = 5',result)
cv2.imshow('Noisy Image',unchanged_noisy_image)
cv2.imshow('Original Image',orignal_image)
cv2.waitKey(0)

'''
5.5 RMSD 0.088584
5 RMSD 0.088554
4.5 RMSD 0.088662
4.0 RMSD 0.088792
RMSD 0.089101 : 3.5
RMSD 0.089881 : 3.0
RMSD 0.091369 : 2.5 
RMSD 0.095336 : 2.0 
RMSD 0.105504 : 1.5 
RMSD 0.108865 : 1.4 
RMSD 0.113178 : 1.3 
RMSD 0.118018 : 1.2 
RMSD 0.124077 : 1.1
RMSD 0.131224 : 1.0
RMSD 0.140126 : 0.9
'''
