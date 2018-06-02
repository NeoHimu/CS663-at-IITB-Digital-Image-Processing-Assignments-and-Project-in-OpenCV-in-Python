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

cv2.imshow('Original Image',img)

# Selecting every dth Pixel
d=int(sys.argv[2])
#print(img.shape)
#print(type(img))
#b,g,r = cv2.split(img) # Split the image into its component chanel. But this method of deletion is quite slow
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

b_cols_got_deleted = np.delete(b, np.s_[::d], 1)
b_rows_and_cols_got_deleted = np.delete(b_cols_got_deleted, np.s_[::d], 0)

g_cols_got_deleted = np.delete(g, np.s_[::d], 1)
g_rows_and_cols_got_deleted = np.delete(g_cols_got_deleted, np.s_[::d], 0)

r_cols_got_deleted = np.delete(r, np.s_[::d], 1)
r_rows_and_cols_got_deleted = np.delete(r_cols_got_deleted, np.s_[::d], 0)

img_d_equals_2 = cv2.merge((b_rows_and_cols_got_deleted,g_rows_and_cols_got_deleted,r_rows_and_cols_got_deleted))

cv2.imshow('image_d_equals_'+str(d),img_d_equals_2)

cv2.waitKey(0)
