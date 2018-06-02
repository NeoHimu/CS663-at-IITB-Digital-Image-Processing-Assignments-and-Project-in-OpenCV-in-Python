import scipy.io
import cv2
import numpy as np
import sys,os


script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../../CroppedYale/')

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

# -------------------------- Top 25 eigen vectors ----------------------
'''
(a,b) = shape_of_image
for counter, v in enumerate(V.T, 1):
	if counter < 26:
		cv2.imshow('Eigen Faces'+str(counter)+'.png',v.reshape((a,b)))
		cv2.waitKey(0) & 0xFF
'''
(a,b) = shape_of_image
if int(sys.argv[1]) == 1000: # 1000 represents eigenvector here
	#print(V.T[int(sys.argv[2])].reshape((a,b)))
	cv2.imshow('Eigen Face '+str(sys.argv[2]),im2double(V.T[int(sys.argv[2])].reshape((a,b))))# sys.argv[2] : eigen vector nuumber
	cv2.waitKey(0) & 0xFF

else:
	#ks = [2, 10, 20, 50, 75, 100, 125, 150, 175]
	ks = [int(sys.argv[1])]
	for k in ks:
		alpha = V.T.dot(X)
		alpha_1 = alpha.T[0] # First column of alpha corresponds to first image

		x_1 = x_bar + V[:,0:k].dot(alpha_1[0:k])

		#first image
		#print(x_1.shape)
	
		cv2.imshow('Reconstructed Image_'+str(k),x_1.reshape((a,b)))
		cv2.waitKey(0) & 0xFF 

