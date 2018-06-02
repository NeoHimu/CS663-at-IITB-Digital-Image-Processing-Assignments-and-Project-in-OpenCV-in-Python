import os, sys
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

#Method for creating a floating point image pixels
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
#----------------------------------------------------------------------------------------------------------------------------
def create_font(fontname='', fontsize=10):
	return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
	fig = plt.figure()
	# main title
	fig.text(.5, .95, title, horizontalalignment='center') 
	for i in xrange(len(images)):
		ax0 = fig.add_subplot(rows,cols,(i+1))
		plt.setp(ax0.get_xticklabels(), visible=False)
		plt.setp(ax0.get_yticklabels(), visible=False)
		if len(sptitles) == len(images):
			plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('',10))
		else:
			plt.title("%s #%d" % (sptitle, (i+1)), create_font('',10))
		plt.imshow(np.asarray(images[i]), cmap=colormap)
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)
		
def imsave(image, title="", filename=None):
	plt.figure()
	plt.imshow(np.asarray(image))
	plt.title(title, create_font('',10))
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)

#----------------------------------------------------------------------------------------------------------------------------

def normalize(X, low, high, dtype=None):
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].	
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

def read_images_att(path, sz=None):
	total_classes = 0
	X,y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			subject_path = os.path.join(dirname, subdirname)
			for filename in os.listdir(subject_path):
				try:
					im = Image.open(os.path.join(subject_path, filename))
					im = im.convert("L")
					# resize to given size (if given)
					if (sz is not None):
						im = im.resize(sz, Image.ANTIALIAS)
					X.append(np.asarray(im, dtype=np.uint8))
					y.append(total_classes)
				except IOError:
					print "I/O error({0}): {1}".format(errno, strerror)
				except:
					print "Unexpected error:", sys.exc_info()[0]
					raise
			total_classes = total_classes+1
	return (X,y)

def read_images_yale(path, sz=None):
	total_classes = 0
	X,y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			count = 0
			subject_path = os.path.join(dirname, subdirname)
			
			for filename in os.listdir(subject_path):
				count += 1
				if count>60:
					break
				try:
					im = Image.open(os.path.join(subject_path, filename))
					im = im.convert("L")
					# resize to given size (if given)
					if (sz is not None):
						im = im.resize(sz, Image.ANTIALIAS)
					X.append(np.asarray(im, dtype=np.uint8))
					y.append(total_classes)
				except IOError:
					print "I/O error({0}): {1}".format(errno, strerror)
				except:
					print "Unexpected error:", sys.exc_info()[0]
					raise
			total_classes = total_classes+1
	return (X,y)


def intoMatrix(X):
	matrix = np.empty((0, X[0].size), dtype=X[0].dtype)
	for x in X:
		matrix = np.vstack((matrix, np.asarray(x).reshape(1,-1)))
	return matrix

def projectionOnSubspace(eigen_vectors, X, mean_X):
	return np.dot(X - mean_X, eigen_vectors)

def reconstruction(eigen_vectors, Y, mean_X):
	return np.dot(Y, eigen_vectors.T) + mean_X

def pca(X, y, kk=0):
	(n,d) = X.shape
	if (kk <= 0) or (kk>n):
		kk = n
	mean_X = X.mean(axis=0)
	X = X - mean_X
	if n>d:
		C = np.dot(X.T,X)
		(eigenvalues,eigenvectors) = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		(eigenvalues,eigenvectors) = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			#unit normalize all eigen vectors
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])

	# sort eigenvectors into descending order by their eigenvalue
	indices = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[indices]
	eigenvectors = eigenvectors[:,indices]
	# select only kk
	eigenvalues = eigenvalues[0:kk].copy()
	eigenvectors = eigenvectors[:,0:kk].copy()
	return (eigenvalues, eigenvectors, mean_X)

def lda(X, y, kk=0):
	y = np.asarray(y)
	(n,d) = X.shape # n : number of images, d : vectorized form of the image
	#print("%d %d"%(n,d))
	#print(len(y))
	total_classes = np.unique(y)
	if (kk <= 0) or (kk>(len(total_classes)-1)):
		kk = len(total_classes)-1
	meanTotal = X.mean(axis=0)
	Sw = np.zeros((d, d), dtype=np.float32)
	Sb = np.zeros((d, d), dtype=np.float32)
	for i in total_classes:
		Xi = X[np.where(y==i)[0],:]
		meanClass = Xi.mean(axis=0)
		Sw = Sw + np.dot((Xi-meanClass).T, (Xi-meanClass)) # rank of this matrix is atmost n-total_classes
		Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))#rank of this matrix is atmost totalClass-1
	eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
	# sort eigenvectors into descending order by their eigenvalue
	indices = np.argsort(-eigenvalues.real)
	eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:,indices]
	# select only kk
	eigenvalues = np.array(eigenvalues[0:kk].real, dtype=np.float32, copy=True)
	eigenvectors = np.array(eigenvectors[:,0:kk].real, dtype=np.float32, copy=True)
	return (eigenvalues, eigenvectors)

def fisherfaces(X,y,kk=0):
	y = np.asarray(y)
	(n,d) = X.shape
	total_classes = len(np.unique(y))
	(eigenvalues_pca, eigenvectors_pca, mean_pca) = pca(X, y, (n-total_classes))
	(eigenvalues_lda, eigenvectors_lda) = lda(projectionOnSubspace(eigenvectors_pca, X, mean_pca), y, kk)
	eigenvectors = np.dot(eigenvectors_pca,eigenvectors_lda)
	return (eigenvalues_lda, eigenvectors, mean_pca)


def eucDist (a, b):
	return np.sqrt(np.sum(np.power((np.asarray(a).flatten()-np.asarray(b).flatten()),2)))

projections = []

def predict(X, y):
	minDist = np.finfo('float').max
	minClass = -1
	Q = projectionOnSubspace(W, X.reshape(1,-1), mean_temp)
	for i in xrange(len(projections)):
		dist = eucDist(projections[i], Q)
		if dist < minDist:
			minDist = dist
			minClass = y[i]
	return minClass


if __name__ == '__main__':
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    #divide into train and test set
    if sys.argv[2] == "1": # att face database 
    	# read images
    	[X,y] = read_images_att(sys.argv[1]) #y stores labels
    	#print(y)
    	for count, x in enumerate(X):
    		#first 32 images only
    		if (count % 10) < 6 and count<320:
    			train_x.append(x)
    			train_y.append(y[count])
    		elif count<320:
    			test_x.append(x)
    			test_y.append(y[count])
    elif sys.argv[2] == "2": #"2" : Yale face database
    	# read images
    	[X,y] = read_images_yale(sys.argv[1]) #y stores labels
    	#print(y)
    	for count, x in enumerate(X):
    		#first 32 images only
    		if (count % 60) < 40:
    			train_x.append(x)
    			train_y.append(y[count])
    		else:
    			test_x.append(x)
    			test_y.append(y[count])
    			
    kks = [37]#[5, 10, 15, 20, 25, 30, 35, 37] # for Yale, number of classes are 38, so it is up to 37
    prediction_rate = []
    for kk in kks:
		# compute the fisherfaces model
		(eig_val, W, mean_temp) = fisherfaces(intoMatrix(train_x),train_y, kk)
		# store projections
		projections = []
		for xi in train_x:
			projections.append(projectionOnSubspace(W, xi.reshape(1,-1), mean_temp))

		total_count = 0
		correct_count = 0
		#print(test_y)
		for count, x in enumerate(test_x):
			#print(predict(x))
			#print(test_y[count])
			total_count += 1
			if test_y[count] == predict(x, train_y):
				correct_count += 1
		
		
		print("Prediction Accuracy %f"%(float(correct_count)/float(total_count)))
		prediction_rate.append(float(correct_count)/float(total_count))

    eigenfaces = []
    for i in xrange(min(W.shape[1], 20)):
    	e = W[:,i].reshape(X[0].shape)
    	eigenfaces.append(normalize(e,0,255))
    	
    # plot
    subplot(title="Fisherfaces", images=eigenfaces, rows=5, cols=4, sptitle="Fisherface", filename="fisherfaces.png")
    script_dir = sys.path[0] # This gives the directory in which our script is running on the system
    img_path = os.path.join(script_dir, "fisherfaces.png")
    img = cv2.imread(img_path) # Read image here
    
    #img = im2double(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    #Reconstruction of first image
    
    eigen_vec = W[:,0].reshape(-1,1)
    projected_image = projectionOnSubspace(eigen_vec, X[0].reshape(1,-1), mean_temp)
    reconstructed_image = reconstruction(eigen_vec, projected_image, mean_temp)
    reconstructed_image = reconstructed_image.reshape(X[0].shape)
    """
    plt.plot(prediction_rate, kks, 'ro')
    plt.axis([0, 1, 0, 40]) # range of x and y axis
    plt.xlabel('Prediction Rate')
    plt.ylabel('Values of k')
    plt.show()
    """
    cv2.imshow("Reconstructed Image Fisherfaces",im2double(reconstructed_image))
    cv2.imshow("Fisherfaces",img)
    cv2.waitKey(0)
