import scipy.io
import cv2
import numpy as np
import sys,os
import matplotlib.pyplot as plt


script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, 'att_faces/')

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

xs = [] # This list will have 6 images for first 32 persons.
shape = (0,0)
for idx1 in range(1,33):
	for idx2 in range(1,7):
		temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
		img = cv2.imread(temp_path) # Read image here
		shape_of_image = np.shape(img[:,:,0])
		xs.append(im2double(img[:,:,0]).ravel())

#print(len(xs))
(a,b) = shape_of_image

x_bar = np.zeros(a*b)

for x in xs:
	x_bar = np.add(x_bar, x)

#Step 1: Compute the mean of the given points.

x_bar = x_bar/float(len(xs))
#print(x_bar)

# Step 2 : Mean deducted Points
xs_bar = []
for x in xs:
	xs_bar.append(x - x_bar)

#print(len(xs_bar))

# Step 3: Computing the L matrix
X = np.column_stack((xs_bar))
#print(X.shape)

L = np.dot(X.T,X)
#print(L.shape)

#Step 4: Compute eigen values and eigen vectors of L
eigvals, eigvecs = np.linalg.eigh(L)
	
print(eigvecs.shape)
# Eigen vectors are arranged according to the increasing eigen values. We'll have to make it according to the decreasing eigen values in order to select k eigen vectors corresponding to k largest eigen values.

eigvecs = eigvecs.T[::-1].T

#Step 5 : Eigen vectors of C from those of L
V = np.dot(X, eigvecs)

#Step 6 : Unit normalize columns of V : Orthonoraml eigenvectors
(a,b) = V.shape
print("Shape of V : ",(V.shape))
for idx3 in range(b):
	V.T[idx3] = V.T[idx3]/ float(sum(V.T[idx3]**2)**0.5)

#print(V)

# Step 7 : alpha_i = V.X_bar_i
alpha = V.T.dot(X)
print("Shape of alpha ",alpha.shape)
alpha_1 = alpha.T[0]


# Test images

xs_test = [] # This list will have last 4 images for first 32 persons.
for idx1 in range(1, 33):
	for idx2 in range(7,11):
		temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
		img = cv2.imread(temp_path) # Read image here
		xs_test.append(im2double(img[:,:,0]).ravel())

# You deduct the mean image from each test image
mean_deducted_test = []
for x_test in xs_test:
	mean_deducted_test.append(x_test-x_bar)
#print(len(mean_deducted_test))

X_test = np.column_stack((mean_deducted_test))
#print(X_test.shape)
alpha_test = V.T.dot(X_test)
print("alpha_test shape ",alpha_test.shape)
# Project each image on eigen-sapce and find the closest image in terms of minimum squared difference of alpha_test and other alphas.

ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170]
prediction_rate = [] #for each k

#grouped_train_images_alpha = np.hsplit(alpha, 32) #32 => number of groups. Here each group size is 6
#print(len(grouped_train_images_alpha))
#print("Train Shape of each group ",grouped_train_images_alpha[0].shape)

#grouped_test_images_alpha = np.hsplit(alpha_test, 32) #32 => number of groups. Here each group size is 4
#print(len(grouped_train_images_alpha))
#print("Test Shape of each group ",grouped_test_images_alpha[0].shape)

#pick one group from the test set and compare each element of it with the other group
for k in ks:
	correct_prediction_count = 0
	for counter, ele in enumerate(np.hsplit(alpha_test, 128),0):
		temp_diff_alphas = []
		for i, a in enumerate(np.hsplit(alpha, 192), 0):
			temp_diff_alphas.append(sum( (alpha.T[i][0:k] - alpha_test.T[counter][0:k])**2))
		index = np.argmin(temp_diff_alphas)
		if (counter/4 == index/6):
			correct_prediction_count += 1

	print(correct_prediction_count)
	prediction_rate.append(correct_prediction_count/float(128))

print(prediction_rate)

k = 150
alpha = V.T.dot(X)
alpha_1 = alpha.T[0] # First column of alpha corresponds to first image

x_1 = x_bar + V[:,0:k].dot(alpha_1[0:k])

#first image
print(x_1.shape)
(a,b) = shape_of_image
cv2.imshow('Image',(x_1.reshape((a,b))))


plt.plot(prediction_rate, ks, 'ro')
plt.axis([0, 1, 0, 200]) # range of x and y axis
plt.xlabel('Prediction Rate')
plt.ylabel('Values of k')
plt.show()

cv2.waitKey(0)
