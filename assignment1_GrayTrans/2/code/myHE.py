import cv2
import numpy as np
import sys,os
from matplotlib import pyplot as plt
import numpy.ma as ma

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])
img = cv2.imread(img_path) # Read image here

color = ('b','g','r')
for i,col in enumerate(color):
	histr = cv2.calcHist([img],[i],None,[256],[0,256])
	plt.plot(histr,color = col)
	plt.xlim([0,256])
#plt.show()


cv2.imshow('Original Image',img)


b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
color = [b,g,r]
# np.histogram : It takes two mandatory parameters here : 1.) Array on which histogram is to be calculated 2.) Total number of bins

histogram,bins = np.histogram(b.flatten(),256)
#print(histogram)
#print(len(histogram))
#histogram.cumsum() is used to generate a cumulative sum of the histogram contents.
cdf = histogram.cumsum()

#print(cdf)
#masked_equal() masks an array where it is equal to a given value i.e. 0 in this case. It will eleminate the invalid 		value. Masked entries are not used in computations. Now we find the minimum histogram value (excluding 0) and apply the 	histogram equalization. To do this, I have used here, the masked array concept array from Numpy. For masked array, all 		operations are performed on non-masked elements. This is done to ignore first few intensities because they might be 	 	outliers.
cdf_masked = ma.masked_equal(cdf,0)
#print(cdf_masked)
#Normalization of the cdf is done as cdf can cross 255 range
cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
#print(cdf_masked)
#missing value is filled with 0. As a general rule, where a representation of the array is required without any masked 		entries, it is recommended to fill the array with the filled method with a suitable value.
cdf = ma.filled(cdf_masked,0).astype('uint8')
#Now we have a mapping from original image intensities to the transformed intensities
hist_equalized_img = cdf[b]
result_array = []
result_array.append(hist_equalized_img)

if np.array_equal(b,g) and np.array_equal(g,r):
	print("Grey")
	cv2.imshow('Histogram_equalized Image',hist_equalized_img)

else:
	print("Color")
	for i in range(1,3):
		histogram,bins = np.histogram(color[i].flatten(),256)
		cdf = histogram.cumsum()
		cdf_masked = ma.masked_equal(cdf,0)
		cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
		cdf = ma.filled(cdf_masked,0).astype('uint8')
		result_array.append(cdf[color[i]])
	cv2.imshow('Histogram_equalized Image',cv2.merge((result_array[0],result_array[1],result_array[2])))


cv2.waitKey(0)
