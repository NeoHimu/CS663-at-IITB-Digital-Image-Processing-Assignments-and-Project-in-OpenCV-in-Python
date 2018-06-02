import scipy.io
import cv2
import numpy as np
import sys,os
import matplotlib.pyplot as plt
from copy import deepcopy 

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/barbara256-part.png')
img = cv2.imread(img_path, 0) # 0 : Gray image
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


#img = im2double(img)

row,col = img.shape
mean = 0
sigma = 20
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
#print(gauss)
noisy = img + gauss

newimg = deepcopy(noisy)#cv2.resize(noisy,(row,col))

patch_size = 3 # Actually it is 7X7

window_size = patch_size
padded = np.concatenate((np.concatenate((np.zeros((row,window_size)), newimg),axis=1), np.zeros((row,window_size))),axis=1)

padded = np.concatenate((np.concatenate((np.zeros((window_size,col+2*(window_size))),padded),axis=0), np.zeros((window_size, col+2*(window_size)))),axis=0)
r,c = padded.shape
result = np.zeros((r,c))
divide = np.zeros((r,c))
q_ones = np.ones((7,7))
#----------------------------- padded noisy image is created------------------------------------
P_list = []

for idx1 in range(window_size,r-window_size):
	for idx2 in range(window_size,c-window_size):
		q_ref = padded[idx1-patch_size:idx1+patch_size+1, idx2-patch_size:idx2+patch_size+1]
		P_list.append(q_ref.ravel())
		
		
P = np.column_stack((P_list))
#Computing eigen vectors of PP.T
eigvals, eigvecs = np.linalg.eigh(np.dot(P, P.T))
eigvecs = eigvecs.T[::-1].T 
V =  eigvecs# V contains eigen vectors of PP.T

#alpha : eigen coefficients
alpha = V.T.dot(P)
print(alpha.shape)
(a,N) = alpha.shape
average_squared_alpha = []

for jth_eigen_coefficient in alpha:
	average_squared_alpha.append(max(0, sum(jth_eigen_coefficient**2)/N-sigma**2))
	
alpha_bar = np.array(average_squared_alpha)
#print(alpha_bar.shape)
alpha_denoised = np.zeros(alpha.shape)
# ith noisy patch and jth eigen coefficient
# Wiener filter update
for i, patch in enumerate(alpha.T):
	for j, ele in enumerate(patch):
		alpha_denoised[j][i] = alpha[j][i]/ (1 + (sigma**2)/(alpha_bar[j]**2)) if alpha_bar[j]>0  else 0
		
patches_denoised = np.dot(V,alpha_denoised)

print(patches_denoised.shape)
row,col = newimg.shape

index = 0
for idx1 in range(window_size,r-window_size):
	for idx2 in range(window_size, c-window_size):
		result[idx1-window_size:idx1+window_size+1, idx2-window_size:idx2+window_size+1] += patches_denoised.T[index].reshape(7,7)
		divide[idx1-window_size:idx1+window_size+1, idx2-window_size:idx2+window_size+1] += q_ones
		
		index += 1

#take average of these patch values
print(result.shape)
result = result[window_size:r-window_size,window_size:c-window_size]
divide = divide[window_size:r-window_size,window_size:c-window_size]
result = np.divide(result,divide)
mse_original_denoised = sum((img-result).ravel()**2)/(row*col)
mse_original_noisy = sum((img-noisy).ravel()**2)/(row*col)
print("MSD for Rsult and original %f"%(mse_original_denoised))
print("MSD for Rsult and Noisy %f"%(mse_original_noisy))
cv2.imshow("Denoised Image", im2double(result))
cv2.imshow("Noisy Image", im2double(noisy))

cv2.waitKey(0) & 0xFF
