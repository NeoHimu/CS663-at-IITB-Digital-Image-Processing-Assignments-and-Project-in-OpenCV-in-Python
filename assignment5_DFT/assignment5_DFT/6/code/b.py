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

print(img.shape)
(row,col) = img.shape

newimg = noisy


patch_size = 3 # Actually it is 7X7

window_size = 15 # Actually it is 31X31
padded = np.concatenate((np.concatenate((np.zeros((row,window_size)), newimg),axis=1), np.zeros((row,window_size))),axis=1)

padded = np.concatenate((np.concatenate((np.zeros((window_size,col+2*(window_size))),padded),axis=0), np.zeros((window_size, col+2*(window_size)))),axis=0)
r,c = padded.shape
result = np.zeros((r,c))
q_ones = np.ones((7,7))
#print(padded.shape)

K=200
count = 0
divide = np.zeros((r,c))
#Consider a small p x p patch - denoted q_ref in noisy image. Top-left of q_ref is at idx1 and idx2
for idx1 in range(window_size,r-window_size):
	for idx2 in range(window_size,c-window_size):
		q_ref = padded[idx1-patch_size:idx1+patch_size+1, idx2-patch_size:idx2+patch_size+1]
		window_ref = padded[idx1-window_size:idx1+window_size+1, idx2-window_size:idx2+window_size+1]
		#Find all patches of size 7X7 in the window of size 31X31
		P_list = []
		for idx3 in range(patch_size,2*window_size+1-patch_size):
			for idx4 in range(patch_size,2*window_size+1-patch_size):
				w = window_ref[idx3-patch_size:idx3+patch_size+1, idx4-patch_size:idx4+patch_size+1]
				#print(w.shape)
				P_list.append(w.ravel())
		
		SErr = []
		P = np.column_stack((P_list))
		for patch in P.T:
			SErr.append(sum((patch-q_ref.ravel())**2))
			
		#Sort P_list based on SErr and select most similar 200 i.e. select first 200 ele in P_list after sorting it in ascending order.
		order_arr = []
		for index in range(len(SErr)):
			order_arr.append(index)
		index_list = [x for _,x in sorted(zip(SErr,order_arr))]
		#print(index_list[0:K])
		Xref = P[:, index_list[0:K]]
		#print(Xref.shape)
		#Step3
		eigvals, eigvecs = np.linalg.eigh(np.dot(Xref, Xref.T))
		#eigvecs = eigvecs.T[::-1].T #eigenvecs are in increaing order. Needs to be sorted in decreasing order.
		V = eigvecs # V contains eigen vectors of PP.T
		#Step4
		alpha = np.dot(V.T, Xref)
		#print(alpha.shape)
		#Step5
		alpha_ref = alpha.T[0]   #This is the first alpha
		
		#Calculating alpha_bar
		(a,N) = alpha.shape
		average_squared_alpha = []
		for jth_eigen_coefficient in alpha:
			average_squared_alpha.append(max(0, sum(jth_eigen_coefficient**2)/N-sigma**2))
		alpha_bar = np.array(average_squared_alpha)
		
		beta_ref_list = []
		for l in range(0,49):
			beta_ref_list.append(alpha_ref[l]/(1 + (sigma**2)/(alpha_bar[l]**2)) if alpha_bar[l]**2>0 else 0)
			
		beta_ref = np.array(beta_ref_list)
		
		#Step 6 => Reconstruct the reference patch 
		q_ref_denoised = np.dot(V, beta_ref)
		#print(q_ref_denoised.shape)
		
		result[idx1-patch_size:idx1+patch_size+1, idx2-patch_size:idx2+patch_size+1] += q_ref_denoised.reshape(7,7)
		divide[idx1-patch_size:idx1+patch_size+1, idx2-patch_size:idx2+patch_size+1] += q_ones
		count += 1
		if count%1000 == 0:
			print(count)
				
		
#take average of these patch values
#temp = result/49
result = result[window_size:r-window_size,window_size:c-window_size]
divide = divide[window_size:r-window_size,window_size:c-window_size]
result = np.divide(result,divide)
mse = sum((img-result).ravel()**2)/(row*col)
print("MSE between Original and Result %f"%(mse))
cv2.imshow("Denoised Image", im2double(result))
cv2.imshow("Noisy Image", im2double(noisy))
#cv2.imshow("Temp Image", im2double(temp))
cv2.waitKey(0) & 0xFF
