import scipy.io
import cv2
import numpy as np
import sys,os
import h5py


script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])

arrays = {}
f = h5py.File(img_path)
for k, v in f.items():
    arrays[k] = np.array(v)

#print(arrays['imageOrig'])
img = np.array(arrays['imageOrig']).T
min_val=0.0
max_val=0.0
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return max_val,min_val,out
max_val,min_val,img = im2double(img)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
Ix = cv2.GaussianBlur(sobelx, (5,5),10.0) #15
Iy = cv2.GaussianBlur(sobely, (5,5), 3.0) #4
Ix2 = np.multiply(Ix,Ix)
Iy2 = np.multiply(Iy,Iy)
IxIy = np.multiply(Ix,Iy)

min_eigen = np.zeros(img.shape)
max_eigen = np.zeros(img.shape)
row,col = img.shape
result = np.zeros(img.shape)
k=0.05
count = 0
for idx1 in range(1,row-1):
	for idx2 in range(1,col-1):
		Ix2_temp = sum((Ix2[idx1-1:idx1+1,idx2-1:idx2+1]).ravel())
		Iy2_temp = sum((Iy2[idx1-1:idx1+1,idx2-1:idx2+1]).ravel())
		IxIy_temp = sum((IxIy[idx1-1:idx1+1,idx2-1:idx2+1]).ravel())
		pixelwise_matrix = [Ix2_temp,IxIy_temp,IxIy_temp,Iy2_temp]
		pixelwise_matrix = np.reshape(pixelwise_matrix,(2,2))
		eigen_values = np.linalg.eigvals(np.array(pixelwise_matrix))
		min_eigen[idx1][idx2] = min(eigen_values[0],eigen_values[1])
		max_eigen[idx1][idx2] = max(eigen_values[0],eigen_values[1])
		
		result[idx1][idx2] = min_eigen[idx1][idx2]*max_eigen[idx1][idx2] - k*(min_eigen[idx1][idx2]+max_eigen[idx1][idx2])**2
		result[idx1][idx2] = result[idx1][idx2] if result[idx1][idx2]>0.03 else min_val #0.001
		if count%10000==0:
			print(count)
		count+=1


result = result*(max_val-min_val)
cv2.imshow("Min eigen value image",min_eigen)
cv2.imshow("Max eigen value image",max_eigen)
cv2.imshow('Ix Image',sobelx)
cv2.imshow('Iy Image',sobely)
cv2.imshow('Superimposed Image',img+result)
cv2.imshow('Original Image',img)
cv2.imshow('Corner Image',result)
cv2.waitKey(0)
