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

print(arrays['imageOrig'])
img = np.array(arrays['imageOrig']).T

gaussian_blur = cv2.GaussianBlur(img, (5,5), 10.0)
print(img.shape)
print(gaussian_blur.shape)
#unsharp_image = cv2.addWeighted(img, 1.8, gaussian_blur, -0.5, 0) #These parameters are suitable for lionCrop.mat
unsharp_image = cv2.addWeighted(img, 1.7, gaussian_blur, -0.5, 0) #These parameters are suitable for superMoonCrop.mat
#print(sum(img-unsharp_image))
cv2.imshow('Unsharped Image',unsharp_image)
cv2.imshow('Original Image',img)
cv2.waitKey(0)
