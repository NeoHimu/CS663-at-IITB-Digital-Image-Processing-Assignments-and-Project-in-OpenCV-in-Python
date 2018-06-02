import cv2
import numpy as np
import sys,os
from matplotlib import pyplot as plt
import numpy.ma as ma
from copy import deepcopy

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])
img = cv2.imread(img_path) # Read image here

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

b_dc = deepcopy(b)
g_dc = deepcopy(g)
r_dc = deepcopy(r)

final_b = deepcopy(b)
final_g = deepcopy(g)
final_r = deepcopy(r)


bgr_color = [b,g,r]

window_size = 150# it is actually 2*150+1 = 301

row,col = b.shape

count = 0

if np.array_equal(b_dc,g_dc) and np.array_equal(g_dc,r_dc):
	print("Grey")
	for i in range(row):
		for j in range(col):
			img = b[(0 if i-window_size<0 else (i-window_size)):row-1 if (i+window_size+1)>row-1 else (i+window_size+1),\
			0 if (j-window_size)<0 else (j-window_size): col-1 if (j+window_size+1)>col-1 else (j+window_size+1)]
			histogram,bins = np.histogram(img.flatten(),256)
			count += 1
			if count%1000==0:
				print(count)
			cdf = histogram.cumsum()
			cdf_masked = ma.masked_equal(cdf,0)
			cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
			cdf = ma.filled(cdf_masked,0).astype('uint8')
			ahe = cdf[b]
			final_b[i][j] = ahe[i][j]
	cv2.imshow('AHE',final_b)
else:
	print("Color")
	for idx in range(0,3):
		for i in range(row):
			for j in range(col):
				img = bgr_color[idx][(0 if i-window_size<0 else (i-window_size)):row-1 if (i+window_size+1)>row-1 \
				else (i+window_size+1),0 if (j-window_size)<0 else (j-window_size): col-1 if (j+window_size+1)>col-1 \
				else (j+window_size+1)]
				histogram,bins = np.histogram(img.flatten(),256)
				count += 1
				if count%1000==0:
					print(count)
				cdf = histogram.cumsum()
				cdf_masked = ma.masked_equal(cdf,0)
				cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
				cdf = ma.filled(cdf_masked,0).astype('uint8')
				if idx==0:
					ahe = cdf[bgr_color[idx]]
					final_b[i][j] = ahe[i][j]
				elif idx==1:
					ahe = cdf[bgr_color[idx]]
					final_g[i][j] = ahe[i][j]
				else:
					ahe = cdf[bgr_color[idx]]
					final_r[i][j] = ahe[i][j]
	
	cv2.imshow('AHE Image',cv2.merge((final_b,final_g,final_r)))

cv2.imshow('Original Image',img)
cv2.waitKey(0)
