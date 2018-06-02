import cv2
import numpy as np
import sys,os
from matplotlib import pyplot as plt
import numpy.ma as ma
import math
from copy import deepcopy

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])
img = cv2.imread(img_path) # Read image here

cv2.imshow('Original Image',img)


b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

b_dc = deepcopy(b)
g_dc = deepcopy(g)
r_dc = deepcopy(r)

final_b = deepcopy(b)
final_g = deepcopy(g)
final_r = deepcopy(r)

color = [b,g,r]

window_size = 150 # it is actually 2*150+1 = 301..

count = 0
row,col = b.shape



if np.array_equal(b_dc,g_dc) and np.array_equal(g_dc,r_dc):
	print("Grey")
	for i in range(row):
		for j in range(col):
	
			img = b[(0 if i-window_size<0 else (i-window_size)):row-1 if (i+window_size+1)>row-1 else (i+window_size+1),\
			0 if (j-window_size)<0 else (j-window_size): col-1 if (j+window_size+1)>col-1 else (j+window_size+1)]

			histogram,bins = np.histogram(img.flatten(),256)
			#print(histogram)
			positive_count=0
			distributable = 0
			#Threshold : maximum allowed frequency of any intensity value in the window of given size
		
			for idx1 in range(256):
				if histogram[idx1]>0:
					positive_count += 1
		
			#Finding the clip level
			top = np.max(histogram)
			bottom = np.min(histogram)
			middle = (top+bottom)/2
			clip_level = middle#top
	
			#print(histogram)
			#print(positive_count)
			for idx3 in range(len(histogram)):
				if histogram[idx3] > clip_level:
					distributable += (histogram[idx3] - clip_level)	
					histogram[idx3] = clip_level
		
			#print(distributable)
			for_each_bin = float(distributable)/positive_count
			#print(for_each_bin)
			for idx4 in range(len(histogram)):
				if histogram[idx4]>0: # distribute among only those intensities which are present in the current window
					histogram[idx4] += math.ceil(for_each_bin)
		
			#print(histogram)
			count += 1
			#print(count)
			if count%1000==0:
				print(count)
			
			cdf = histogram.cumsum()

			cdf_masked = ma.masked_equal(cdf,0)
			# As CDF can go beyond 255, so normalization is done to bring it in the range of 0-255
			cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())

			cdf = ma.filled(cdf_masked,0).astype('uint8')
			#print(cdf)
			#Complete mapping is copied to hist_equalized_img
			clahe = cdf[b]
			# But only [i][j] pixel value i.e. the central pixel is updated in the final result that will be displayed
			final_b[i][j] = clahe[i][j]

	cv2.imshow('CLAHE Image',final_b)
	cv2.waitKey(0)
else:
	print("Color")
	for idx_color in range(0,3):
		for i in range(row):
			for j in range(col):
				img = color[idx_color][(0 if i-window_size<0 else (i-window_size)):row-1 if (i+window_size+1)>row-1 \
				else (i+window_size+1),0 if (j-window_size)<0 else (j-window_size): col-1 if (j+window_size+1)>col-1 \
				else (j+window_size+1)]

				histogram,bins = np.histogram(img.flatten(),256)
				#print(histogram)
				positive_count=0
				distributable = 0
				#Threshold : maximum allowed frequency of any intensity value in the window of given size
				for idx1 in range(256):
					if histogram[idx1]>0:
						positive_count += 1
		
				#Finding the clip level
				top = np.max(histogram)
				bottom = np.min(histogram)
				middle = (top+bottom)/2
				clip_level = middle
				for idx3 in range(len(histogram)):
					if histogram[idx3] > clip_level:
						distributable += (histogram[idx3] - clip_level)	
						histogram[idx3] = clip_level
		
				#print(distributable)
				for_each_bin = float(distributable)/positive_count
				#print(for_each_bin)
				for idx4 in range(len(histogram)):
					if histogram[idx4]>0: # distribute among only those intensities which are present in the current window
						histogram[idx4] += math.ceil(for_each_bin)
				count += 1
				if count%1000==0:
					print(count)
				cdf = histogram.cumsum()
				cdf_masked = ma.masked_equal(cdf,0)
				cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
				cdf = ma.filled(cdf_masked,0).astype('uint8')
				if idx_color == 0:
					hist_equalized_img = cdf[color[idx_color]]
					final_b[i][j] = hist_equalized_img[i][j]
				elif idx_color == 1:
					hist_equalized_img = cdf[color[idx_color]]
					final_g[i][j] = hist_equalized_img[i][j]
				else:
					hist_equalized_img = cdf[color[idx_color]]
					final_r[i][j] = hist_equalized_img[i][j]
		
	cv2.imshow('CLAHE Image',cv2.merge((final_b,final_g,final_r)))
	cv2.waitKey(0)
