import scipy.io
import cv2
import numpy as np
import sys,os


script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../../att_faces/')

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

xs_train_1 = [] # This list will have 5 images for first 32 persons.
xs_validation_1 = [] # This list will have 1 image for first 32 persons.
xs_train_2 = [] # This list will have 5 images for first 32 persons.
xs_validation_2 = [] # This list will have 1 image for first 32 persons.
xs_train_3 = [] # This list will have 5 images for first 32 persons.
xs_validation_3 = [] # This list will have 1 image for first 32 persons.
xs_train_4 = [] # This list will have 5 images for first 32 persons.
xs_validation_4 = [] # This list will have 1 image for first 32 persons.
xs_train_5 = [] # This list will have 5 images for first 32 persons.
xs_validation_5 = [] # This list will have 1 image for first 32 persons.
xs_train_6 = [] # This list will have 5 images for first 32 persons.
xs_validation_6 = [] # This list will have 1 image for first 32 persons.

xs_train = []
xs_test_first_32 = []
shape = (0,0)
for idx1 in range(1,33):
	for idx2 in range(1,7):
		# First 6 images of first 32 people for train set
		temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
		img = cv2.imread(temp_path) # Read image here
		shape_of_image = np.shape(img[:,:,0])
		xs_train.append(im2double(img[:,:,0]).ravel())
	
		if idx2 <= 5:  # 1,2,3,4,5
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_train_1.append(im2double(img[:,:,0]).ravel())
		else:
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_validation_1.append(im2double(img[:,:,0]).ravel())
			
		if idx2 >= 2: # 2,3,4,5,6
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_train_2.append(im2double(img[:,:,0]).ravel())
		else:
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_validation_2.append(im2double(img[:,:,0]).ravel())
			
		if idx2 > 2 or idx2 == 1: # 3,4,5,6,1
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_train_3.append(im2double(img[:,:,0]).ravel())
		else:
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_validation_3.append(im2double(img[:,:,0]).ravel())
			
		if idx2 <= 2 or idx2 > 3: # 4,5,6,1,2
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_train_4.append(im2double(img[:,:,0]).ravel())
		else:
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_validation_4.append(im2double(img[:,:,0]).ravel())
			
		if idx2 <= 3 or idx2 > 4: # 5,6,1,2,3
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_train_5.append(im2double(img[:,:,0]).ravel())
		else:
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_validation_5.append(im2double(img[:,:,0]).ravel())
			
		if idx2 <= 4 or idx2 > 5: # 6,1,2,3,4
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_train_6.append(im2double(img[:,:,0]).ravel())
		else:
			temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
			img = cv2.imread(temp_path) # Read image here
			shape_of_image = np.shape(img[:,:,0])
			xs_validation_6.append(im2double(img[:,:,0]).ravel())

		
	# Last 4 images of first 32 people for test set
	for idx2 in range(7,11):
		temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
		img = cv2.imread(temp_path) # Read image here
		shape_of_image = np.shape(img[:,:,0])
		xs_test_first_32.append(im2double(img[:,:,0]).ravel())
	

print(len(xs_test_first_32))	
print(len(xs_train))		

#print(len(xs_train_1))
#print(len(xs_validation_1))
#print(len(xs_train_2))
#print(len(xs_validation_2))
#print(len(xs_train_3))
#print(len(xs_validation_3))



def findThreshold (xs_train, xs_validation, k):
	threshold_person_specific = []
	(a,b) = shape_of_image
	x_bar = np.zeros(a*b)
	for x in xs_train:
		x_bar = np.add(x_bar, x)

	#Step 1: Compute the mean of the given points.
	x_bar = x_bar/float(len(xs_train))
	#print(x_bar)

	# Step 2 : Mean deducted Points
	xs_bar = []
	for x in xs_train:
		xs_bar.append(x - x_bar)
	#print(len(xs_bar))

	# Step 3: Computing the L matrix
	X = np.column_stack((xs_bar))
	#print(X.shape)

	L = np.dot(X.T,X)
	#print(L.shape)

	#Step 4: Compute eigen values and eigen vectors of L
	eigvals, eigvecs = np.linalg.eigh(L)
	
	#print(eigvecs.shape)
	# Eigen vectors are arranged according to the increasing eigen values. We'll have to make it according to the decreasing eigen values in order to select k eigen vectors corresponding to k largest eigen values.

	eigvecs = eigvecs.T[::-1].T

	#Step 5 : Eigen vectors of C from those of L
	V = np.dot(X, eigvecs)

	#Step 6 : Unit normalize columns of V : Orthonoraml eigenvectors
	(a,b) = V.shape
	#print("Shape of V : ",(V.shape))
	for idx3 in range(b):
		V.T[idx3] = V.T[idx3]/ float(sum(V.T[idx3]**2)**0.5)

	#print(V)

	# Step 7 : alpha_i = V.X_bar_i
	alpha = V.T.dot(X)
	#print("Shape of alpha ",alpha.shape)
	alpha_1 = alpha.T[0]


	# You deduct the mean image from each validation image
	mean_deducted_validation = []
	for x_validation in xs_validation:
		mean_deducted_validation.append(x_validation-x_bar)
	#print(len(mean_deducted_test))

	X_validation = np.column_stack((mean_deducted_validation))
	#print(X_test.shape)
	alpha_validation = V.T.dot(X_validation)
	#print("alpha_test shape ",alpha_validation.shape)
	# Project each image on eigen-sapce and find the closest image in terms of minimum squared difference of alpha_test and other alphas.
	ks = [k]
	#ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170]
	
	#pick one group from the validation set and compare each element of it with the other group
	max_error_on_matched_identity = 0
	for k in ks:
		for counter, ele in enumerate(np.hsplit(alpha_validation, 32),0): # an array of array of single vector
			temp_diff_alphas = []
			for i, a in enumerate(np.hsplit(alpha, 160), 0): #total 128 partitions of array containing 128 images
				temp_diff_alphas.append(sum( (alpha.T[i][0:k] - alpha_validation.T[counter][0:k])**2))
			index = np.argmin(temp_diff_alphas)
			temp_error = sum( (alpha.T[index][0:k] - alpha_validation.T[counter][0:k])**2)
			
			#correct prediction
			if (counter == index/5):
				#max_error_on_matched_identity = max(max_error_on_matched_identity, temp_error)
				threshold_person_specific.append(temp_error) 
			else: # incorrect prediction => 0 error.. so that this error is excluded for prediction
				threshold_person_specific.append(0) 
	'''
	threshold_p = []
	for counter, ele in enumerate(threshold_person_specific):
		if counter%2 == 1:
			threshold_p.append(max (threshold_person_specific[counter-1], threshold_person_specific[counter]))
	'''
	return threshold_person_specific#threshold_p #max_error_on_matched_identity #threshold for this train-validation set


# Test images
xs_test_last_8 = [] # This list will have 10 images for last 8 persons.
for idx1 in range(33, 41):
	for idx2 in range(1,11):
		temp_path = img_path+"s"+str(idx1)+"/"+str(idx2)+".pgm"
		img = cv2.imread(temp_path) # Read image here
		xs_test_last_8.append(im2double(img[:,:,0]).ravel())

def prediction ():
	(a,b) = shape_of_image
	x_bar = np.zeros(a*b)
	for x in xs_train:
		x_bar = np.add(x_bar, x)

	#Step 1: Compute the mean of the given points.
	x_bar = x_bar/float(len(xs_train))
	#print(x_bar)

	# Step 2 : Mean deducted Points
	xs_bar = []
	for x in xs_train:
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
	#print("Shape of V : ",(V.shape))
	for idx3 in range(b):
		V.T[idx3] = V.T[idx3]/ float(sum(V.T[idx3]**2)**0.5)

	#print(V)

	# Step 7 : alpha_i = V.X_bar_i
	alpha = V.T.dot(X)
	#print("Shape of alpha ",alpha.shape)
	alpha_1 = alpha.T[0]

	#------------------------------------------------ First 128 images from a first 32 people ----------------------------
	# You deduct the mean image from each test image
	mean_deducted_first_32 = []
	for x_32 in xs_test_first_32:
		mean_deducted_first_32.append(x_32-x_bar)
	#print(len(mean_deducted_test))

	X_first_32 = np.column_stack((mean_deducted_first_32))
	#print(X_test.shape)
	alpha_first_32 = V.T.dot(X_first_32)
	#print("alpha_first_32 shape ",alpha_first_32.shape)
	
	
	# You deduct the mean image from each test image
	mean_deducted_last_8 = []
	for x_8 in xs_test_last_8:
		mean_deducted_last_8.append(x_8-x_bar)
	#print(len(mean_deducted_test))

	X_last_8 = np.column_stack((mean_deducted_last_8))
	#print(X_test.shape)
	alpha_last_8 = V.T.dot(X_last_8)
	#print("alpha_last_8 shape ",alpha_last_8.shape)
	
	
	
	# Project each image on eigen-sapce and find the closest image in terms of minimum squared difference of alpha_test and other alphas.
	
	
	# Project each image on eigen-sapce and find the closest image in terms of minimum squared difference of alpha_test and other alphas.
	ks = [1, 2, 3, 5, 10, 15, 20,23,25,27,29, 30, 50, 75, 100, 150, 170]
	#ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170]
	
	#pick one group from the validation set and compare each element of it with the other group
	max_error_on_matched_identity = 0
	for k in ks:
		correct_prediction = 0
		total_prediction = 0
		false_negative = 0
		l1 = findThreshold (xs_train_1, xs_validation_1, k)
		l2 = findThreshold (xs_train_2, xs_validation_2, k)
		l3 = findThreshold (xs_train_3, xs_validation_3, k)
		l4 = findThreshold (xs_train_4, xs_validation_4, k)
		l5 = findThreshold (xs_train_5, xs_validation_5, k)
		l6 = findThreshold (xs_train_6, xs_validation_6, k)
		threshold_personwise = []
		for counter, ele in enumerate(l1):
			threshold_personwise.append(max(l1[counter] ,max(l2[counter], max(l3[counter], max(l4[counter], max(l5[counter],l6[counter]))))))
		#print(len(l1))
		#print(threshold_personwise)
		#print(len(threshold_personwise))
		#print(threshold)
		
		for counter, ele in enumerate(np.hsplit(alpha_first_32, 128),0): # an array of array of single vector
			temp_diff_alphas = []
			for i, a in enumerate(np.hsplit(alpha, 192), 0): #total 192 partitions of array containing 192 images
				temp_diff_alphas.append(sum( (alpha.T[i][0:k] - alpha_first_32.T[counter][0:k])**2))
			index = np.argmin(temp_diff_alphas)
			temp_error = sum( (alpha.T[index][0:k] - alpha_first_32.T[counter][0:k])**2)
			
			#total prediction
			#if (temp_error < threshold_personwise[counter/4]):
			#	total_prediction += 1
				
			#correct prediction
			if (counter/4 == index/6 and temp_error < threshold_personwise[counter/4]):
				correct_prediction += 1
			
			
	
	#------------------------------------- Last 80 images from 8 people ---------------------------------------
	
		correct_prediction_last_80_only = 0
		false_negative_last_80_only = 0
		#pick one group from the validation set and compare each element of it with the other group
		for counter, ele in enumerate(np.hsplit(alpha_last_8, 80),0): # an array of array of single vector
			temp_diff_alphas = []
			for i, a in enumerate(np.hsplit(alpha, 192), 0): #total 192 partitions of array containing 192 images
				temp_diff_alphas.append(sum( (alpha.T[i][0:k] - alpha_last_8.T[counter][0:k])**2))
			index = np.argmin(temp_diff_alphas)
			temp_error = sum( (alpha.T[index][0:k] - alpha_last_8.T[counter][0:k])**2)
			
			#correct prediction is incremented by 1
			if (temp_error > threshold_personwise[index/6]):
				correct_prediction_last_80_only += 1
			
				
			
		print("k = %d, false_negative = %d out of 208 test images"%(k,128-correct_prediction))
		print("k = %d, correct prediction = %d out of 208 test images"%(k,correct_prediction+correct_prediction_last_80_only))
		print("k = %d, correct prediction = %d out of 80 test images"%(k, correct_prediction_last_80_only))
		print("")

prediction ()

