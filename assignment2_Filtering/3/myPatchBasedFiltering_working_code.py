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

#Resizing and blurring
#img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
#img = cv2.GaussianBlur(img, (5,5), 0.66)


row,col = img.shape
mean = 0
sigma = 0.05# standard deviation is 5% of the intensity range and intensity range is [0-1]
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
#print(gauss)
noisy = img + gauss
unchanged_noisy_image = deepcopy(noisy)
window_size = 14 # Actually window size is 2*12+1 = 25X25
patch_size = 4 # Actually patch size is 2*4+1 = 9X9
result = np.zeros(noisy.shape)
count = 0
#print("%d %d"%(row,col))

def gaussian(x, sigma):
	return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
	
SD = 0.1 #Standard Deviation

for idx1 in range(row):
	for idx2 in range(col):
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
				denominator += numerator
				final_value_of_p_intensity += numerator*img_temp[idx3][idx4] 
				#numerator_spatial = math.exp(-((center_r-idx3)**2+(center_r-idx4)**2)/SD**2)
				result[idx1][idx2] = float(final_value_of_p_intensity)/denominator
		
				
		
		count += 1
		if count%1000==0:
			print(count)
	
		
	

cv2.imshow('Patch Based Filtered Image SD = 0.05 Resized and blurred',result)
cv2.imshow('Original Noisy Image',unchanged_noisy_image)
cv2.waitKey(0)
