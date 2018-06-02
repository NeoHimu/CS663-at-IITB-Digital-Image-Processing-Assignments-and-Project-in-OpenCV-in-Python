import cv2
import numpy as np
import sys,os
from matplotlib import pyplot as plt
from copy import deepcopy

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])
img = cv2.imread(img_path) # Read image here

#Create histogram before converting into float
color = ('b','g','r')
'''
for i,col in enumerate(color):
	histr = cv2.calcHist([img],[i],None,[256],[0,256])
	plt.plot(histr,color = col)
	plt.xlim([0,256])
plt.show()
'''

#Method for creating a floating point image pixels
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = im#.astype('float')
    return out, min_val, max_val
    
img,min_val,max_val = im2double(img)
#print(img.shape)
print("%f %f"%(min_val,max_val))


b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

min_val=27.0
max_val=235.0
row,col = b.shape
b_ce = deepcopy(b)
g_ce = deepcopy(g)
r_ce = deepcopy(r)

result = img

if np.array_equal(b,g) and np.array_equal(g,r):
	print("Grey")
	for i in range(row):
		for j in range(col):
			x=b[i][j] - min_val
			if b[i][j]<min_val:
				b_ce[i][j]=0
			elif b[i][j]>max_val:
				b_ce[i][j]=255
			else:
				b_ce[i][j] = (x)*((255.0)/(max_val-min_val))
	result = b_ce
else:
	print("Color")
	for i in range(row):
		for j in range(col):
			x=b[i][j] - min_val
			if b[i][j]<min_val:
				b_ce[i][j]=0
			elif b[i][j]>max_val:
				b_ce[i][j]=255
			else:
				b_ce[i][j] = (x)*((255.0)/(max_val-min_val))
	for i in range(row):
		for j in range(col):
			x=g[i][j] - min_val
			if g[i][j]<min_val:
				g_ce[i][j]=0
			elif g[i][j]>max_val:
				g_ce[i][j]=255
			else:
				g_ce[i][j] = (x)*((255.0)/(max_val-min_val))

	for i in range(row):
		for j in range(col):
			x=r[i][j] - min_val
			if r[i][j]<min_val:
				r_ce[i][j]=0
			elif r[i][j]>max_val:
				r_ce[i][j]=255
			else:
				r_ce[i][j] = (x)*((255.0)/(max_val-min_val))
	result = cv2.merge((b_ce,g_ce,r_ce))
'''
color= ('b')
for i,col in enumerate(color):
	histr = cv2.calcHist([b_ce],[i],None,[256],[0,256])
	plt.plot(histr,color = col)
	plt.xlim([0,256])
plt.show()
'''

cv2.imshow('Original Image',img)
cv2.imshow('Linearly Contrast Enhanced Image',result)
cv2.waitKey(0)

