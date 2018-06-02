import scipy.io
import cv2
import numpy as np
import sys,os
import matplotlib.pyplot as plt

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, 'CroppedYale/')

# total number of images to be loaded from the folder
def load_images_from_folder(folder, count1, count2):
    images_train = []
    images_test = []
    
    temp_count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if temp_count<count1:
            images_train.append(img)
            temp_count += 1
        elif temp_count<count2:
        	images_test.append(img)
        	temp_count += 1
        #print(temp_count)
        
    return (images_train,images_test)


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

xs = [] # This list will have 40 images for first 38 persons.
xs_test = [] # This list will have 20 images for first 38 persons
shape_of_image = (0,0)
for idx1 in range(1,40):
	if idx1 != 14:
		conditioned_path = "0"
		if idx1<10:
			conditioned_path = "0"+str(idx1)
		else:
			conditioned_path = str(idx1)
		(images_train,images_test) = load_images_from_folder(img_path+"yaleB"+conditioned_path, 40,60) 
		for img in images_train:
			xs.append(im2double(img[:,:,0]).ravel())
			shape_of_image = np.shape(img[:,:,0])
		for img in images_test:
			xs_test.append(im2double(img[:,:,0]).ravel())
			
		
print("xs length ",len(xs))
print("xs_test length ",len(xs_test))
(a,b) = shape_of_image

x_bar = np.zeros(a*b)

for x in xs:
	x_bar = np.add(x_bar, x)

#Step 1: Compute the mean of the given points.

x_bar = x_bar/float(len(xs))
#print(x_bar)

# Step 2 : Mean decuted Points
xs_bar = []
for x in xs:
	xs_bar.append(x - x_bar)

#print(len(xs_bar))

# Step 3: Computing the C matrix
X = np.column_stack((xs_bar))
#print(X.shape)

#---------------------------------------------------------------------------------------------------------------------
u, s, v = np.linalg.svd(X, full_matrices=False)
#s : The singular values for every matrix, sorted in descending order
#Eigen vectors of C 
V = u
V = V[:, int(sys.argv[1]):] # number of eigenvectors to remove = int(sys.argv[1])
#---------------------------------------------------------------------------------------------------------------------

#Step 6 : Unit normalize columns of V : Orthonoraml eigenvectors
(a,b) = V.shape
print("Shape of V : ",(V.shape))
for idx3 in range(b):
	V.T[idx3] = V.T[idx3]/ float(sum(V.T[idx3]**2)**0.5)

#print(V)

# Step 7 : alpha_i = V.X_bar_i
alpha = V.T.dot(X)
print("Shape of alpha", alpha.shape)
alpha_1 = alpha.T[0]



# Test images

# You deduct the mean image from each test image
mean_deducted_test = []
for x_test in xs_test:
	mean_deducted_test.append(x_test-x_bar)


X_test = np.column_stack((mean_deducted_test))
#print(X_test.shape)
alpha_test = V.T.dot(X_test)
print("alpha_test shape ",alpha_test.shape)
# Project each image on eigen-sapce and find the closest image in terms of minimum squared difference of alpha_test and other alphas.
#--------------------------------- (a) part -------------------------------------------------------------------------------
#ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000]
ks = [1000]
prediction_rate = [] #for each k


#pick one group from the test set and compare each element of it with the other group
for k in ks:
	correct_prediction_count = 0
	for counter, ele in enumerate(np.hsplit(alpha_test, len(xs_test)),0):
		temp_diff_alphas = []
		for i, a in enumerate(np.hsplit(alpha, len(xs)), 0):
			temp_diff_alphas.append(sum( (alpha.T[i][0:k] - alpha_test.T[counter][0:k])**2))
		index = np.argmin(temp_diff_alphas)
		if (counter/20 == index/40):
			correct_prediction_count += 1

	#print(correct_prediction_count)
	prediction_rate.append(correct_prediction_count/float(len(xs_test)))

print(prediction_rate)

k = 1000
alpha = V.T.dot(X)
alpha_1 = alpha.T[0] # First column of alpha corresponds to first image

x_1 = x_bar + V[:,0:k].dot(alpha_1[0:k])

#first image
print(x_1.shape)
(a,b) = shape_of_image
cv2.imshow('Image',(x_1.reshape((a,b))))
cv2.waitKey(0)


plt.plot(prediction_rate, ks, 'ro')
plt.axis([0, 1, 0, 1000]) # range of x and y axis
plt.xlabel('Prediction Rate')
plt.ylabel('Values of k')
plt.show()

'''
[0.04078947368421053, 0.034210526315789476, 0.015789473684210527, 0.031578947368421054, 0.2, 0.3078947368421053, 0.3973684210526316, 0.46710526315789475, 0.5657894736842105, 0.5960526315789474, 0.618421052631579, 0.6355263157894737, 0.6618421052631579, 0.7065789473684211, 0.7171052631578947, 0.718421052631579, 0.7236842105263158]
'''

#----------------------------------- (b) part -----------------------------------------------------------------------------
# the squared difference between all except the three eigencoefficients corresponding to the eigenvectors with the three largest eigenvalues.
'''
V = u

# Step : Remove eigen vectors corresponding to largest three eigen values
V = V[:, 3:]
# Step 7 : alpha_i = V.X_bar_i
alpha = V.T.dot(X)
print("Shape of alpha", alpha.shape)
# Test images

# You deduct the mean image from each test image
mean_deducted_test = []
for x_test in xs_test:
	mean_deducted_test.append(x_test-x_bar)

X_test = np.column_stack((mean_deducted_test))
#print(X_test.shape)
alpha_test = V.T.dot(X_test)
print("alpha_test shape ",alpha_test.shape)
# Project each image on eigen-sapce and find the closest image in terms of minimum squared difference of alpha_test and other alphas.
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000]


prediction_rate = [] #for each k


#pick one group from the test set and compare each element of it with the other group
for k in ks:
	correct_prediction_count = 0
	for counter, ele in enumerate(np.hsplit(alpha_test, len(xs_test)),0):
		temp_diff_alphas = []
		for i, a in enumerate(np.hsplit(alpha, len(xs)), 0):
			temp_diff_alphas.append(sum( (alpha.T[i][0:k] - alpha_test.T[counter][0:k])**2))
		index = np.argmin(temp_diff_alphas)
		if (counter/20 == index/40):
			correct_prediction_count += 1

	#print(correct_prediction_count)
	prediction_rate.append(correct_prediction_count/float(len(xs_test)))

print(prediction_rate)


plt.plot(prediction_rate, ks, 'ro')
plt.axis([0, 1, 0, 1000]) # range of x and y axis
plt.xlabel('Prediction Rate: Top three eigen vectors removed')
plt.ylabel('Values of k')
plt.show()
'''
'''
[0.035526315789473684, 0.05789473684210526, 0.14605263157894738, 0.33421052631578946, 0.5460526315789473, 0.6368421052631579, 0.6947368421052632, 0.7578947368421053, 0.8144736842105263, 0.8276315789473684, 0.8315789473684211, 0.8421052631578947, 0.8513157894736842, 0.8618421052631579, 0.8697368421052631, 0.8710526315789474, 0.8736842105263158]
'''

