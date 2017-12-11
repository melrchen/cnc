import numpy as np 
from sklearn import svm
import featurization as ft
import preprocessing as pp

def classifier(X, y):
	'''
	Returns trained model for feature vectors X and binary classes y.

	Args:
		X (np.array): Array of feature vectors of image.
		y (np.array): Binary array corresponding to whether a pixel is
					  a particular color

	Returns:
		Trained model for that particular color
	'''
	# SVM linear classification: particular color or not
	clf = svm.LinearSVC(random_state = 0, multi_class='ovr') # ovr: one vs. rest
	clf.fit(X, y) 

	print('\n')

	return clf


def binary_vector(y, color):
	'''
	Args:
		y (np.array): Array of pixel colors [U,V]
		color (tuple): One vs. rest we're trying to classify

	Returns:
		Binary vector (np.array) with entry 1 if the pixel is that color,
		0 else. 
	'''
	binaryvec = [] # Build binary vector
	for row in y:
		for elt in row:
			if tuple(elt) == color:
				binaryvec.append(1)
			else:
				binaryvec.append(0)

	binaryvec = np.array(binaryvec)

	return binaryvec



def SVM_classification(filename):
	'''
	Fits a set of SVMs, one per discretized color.

	Args:
		filename (string)
	'''
	# Get feature vectors and discretized color for each pixel
	print('=========Preprocessing========')
	Y, X = pp.discretize_colors(filename), ft.featurize(filename)

	# Create copy of Y with entries corresponding to U,V values
	y = []
	for row in Y:
		tmp = []
		for elt in row:
			tmp.append(np.array(elt[1:]))
		y.append(tmp)
	y = np.array(y)

	# Set of clusters (tuple U, V)
	clusters = []
	for row in y:
		for elt in row:
			if tuple(elt) not in clusters:
				clusters.append(tuple(elt))

	# Train an SVM per color
	classifiers = []

	print('=========Training===========')
	for color in clusters:
		yvec = binary_vector(y, color)
		# print(yvec.shape)
		# print(X.shape)
		svmclassifier = classifier(X, yvec)
		print('W: {}'.format(svmclassifier.coef_))
		print('W0: {}'.format(svmclassifier.intercept_))
		classifiers.append(svmclassifier)

	return clusters, classifiers
	



if __name__ == "__main__":
	SVM_classification('doggo.png')