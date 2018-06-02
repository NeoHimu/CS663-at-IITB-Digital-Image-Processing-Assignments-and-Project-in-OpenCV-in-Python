import cv2
import numpy as np
import sys,os

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])
img = cv2.imread(img_path) # Read image here

#Method for creating a floating point image pixels
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
img = im2double(img)
#print(img.shape)
cv2.imshow('Original Image',img)

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
row,col = b.shape
b_temp_cols_expanded = np.insert(np.zeros((2*row,col)), np.s_[::2],b , 0) 
b_temp_cols_and_rows_expanded = np.insert(np.zeros((3*row,col)), np.s_[::1],b_temp_cols_expanded , 1) 
#last col deleted
temp = np.delete(b_temp_cols_and_rows_expanded, 2*row-1, 1)
#last row deleted
b_temp = np.delete(temp, 3*row-1, 0)
#2nd last row deleted
b_final = np.delete(b_temp, 3*row-2, 0)

# 8 neighbours
if sys.argv[1]=="barbaraSmall.png":
	b_final[1] = b_final[0]
	b_final[305] = b_final[306]
	for i in range(2,306):

		if i%3==0:
			b_final[i-1] = b_final[i]
			b_final[i+1] = b_final[i]

	b_final.T[1]= b_final.T[0]
	b_final.T[203] = b_final.T[204]
	for i in range(2,204):
		if i%2==0:
			b_final.T[i-1] = b_final.T[i]
			b_final.T[i+1] = b_final.T[i]
else:
	b_final[1] = b_final[0]
	b_final[3*row-4] = b_final[3*row-3]
	for i in range(2,3*row-4):

		if i%3==0:
			b_final[i-1] = b_final[i]
			b_final[i+1] = b_final[i]

	b_final.T[1]= b_final.T[0]
	b_final.T[2*row-3] = b_final.T[2*row-2]
	for i in range(2,2*row-3):
		if i%2==0:
			b_final.T[i-1] = b_final.T[i]
			b_final.T[i+1] = b_final.T[i]

#print(b_final)

#print(b_final.shape)
cv2.imshow('Nearest Neighbour Enlarged Image',b_final)

cv2.waitKey(0)
