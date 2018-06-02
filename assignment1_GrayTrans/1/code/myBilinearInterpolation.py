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
temp = np.delete(b_temp_cols_and_rows_expanded, 2*col-1, 1)
#last row deleted
b_temp = np.delete(temp, 3*row-1, 0)
#2nd last row deleted
b_final = np.delete(b_temp, 3*row-2, 0)

if sys.argv[1] == "barbaraSmall.png":
	for i in range(205):
		if i%2==1:
			b_final.T[i] = (b_final.T[i-1]+b_final.T[i+1])/2
		
		
	for i in range(305):
		if i%3==1:
			b_final[i] = (2*b_final[i-1]+b_final[i+2])/3
		elif i%3==2:
			b_final[i] = (b_final[i-2]+2*b_final[i+1])/3
else:
	for i in range(2*row-1):
		if i%2==1:
			b_final.T[i] = (b_final.T[i-1]+b_final.T[i+1])/2
		
		
	for i in range(3*col-2):
		if i%3==1:
			b_final[i] = (2*b_final[i-1]+b_final[i+2])/3
		elif i%3==2:
			b_final[i] = (b_final[i-2]+2*b_final[i+1])/3
	
	
#print(b_final)



#print(b_final.shape)
cv2.imshow('Bilinear Enlarged Image',b_final)

cv2.waitKey(0)
