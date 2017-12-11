# import the necessary packages
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
from sklearn.decomposition import PCA

def PCAthatbitch(X):
	'''
	Applies PCA to a matrix of features X and returns the PCA'd X.
	'''
	pca = PCA(n_components = 30)
	Y = pca.fit_transform(X)

	return Y


def featurize(filename):
	'''
	Returns FFT features of an image given by filename.

	Args:
		filename (String)

	Returns:
		feature_vector (np.array): Matrix of feature vectors corresponding to image pixels.
	'''
	# load the image and convert it to a floating point data type
	#image = img_as_float(io.imread(args["image"]))
	image = cv2.imread(filename)
	yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	gray_padded = np.pad(gray, [(7, 7), (7, 7)], mode='constant')

	feature_vector = []

	for ii in range(0, gray.shape[0]):
	    for jj in range(0, gray.shape[1]):
	        pixel = (ii + 7, jj + 7)

	        square = gray_padded[pixel[0]-7 : pixel[0]+8, pixel[1]-7 : pixel[1]+8]
	        #print square.shape
	        #surf = cv2.SURF(.1)
	        #kp, des = surf.detectAndCompute(square,None)
	        features = np.fft.fft2(square).flatten()
	        features = np.append(features, np.mean(square))
	        features = np.append(features, np.std(square))

	        feature_vector.append(features)
	    
	feature_vector = np.array(feature_vector)
	feature_vector = PCAthatbitch(feature_vector)

	return feature_vector
#surf = cv2.SURF(500)
#kp, des = surf.detectAndCompute(image,None)



#img2 = cv2.drawKeypoints(square,kp,None,(255,0,0),4)
#plt.imshow(img2),plt.show()

if __name__ == "__main__":
	print(featurize('doggo.png').shape)
