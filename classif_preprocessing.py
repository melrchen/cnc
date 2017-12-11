import pylab as pl
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
from keras.preprocessing import image
from PIL import Image

# Paths to images: list of tuple (filepath, one-hot encoding)
PLAIN_PATHS = [os.path.join(os.getcwd(), 'plain', file) for file in os.listdir('plain')]
BEACH_PATHS = [os.path.join(os.getcwd(), 'beach', file) for file in os.listdir('beach')]
MTN_PATHS = [os.path.join(os.getcwd(), 'mountain', file) for file in os.listdir('mountain')]
CAVE_PATHS = [os.path.join(os.getcwd(), 'cave', file) for file in os.listdir('cave')]
CITY_PATHS = [os.path.join(os.getcwd(), 'city', file) for file in os.listdir('city')]

# Paths to gray images
PLAIN_PATHS_G = [os.path.join(os.getcwd(), 'plaingray', file) for file in os.listdir('plaingray')]
BEACH_PATHS_G = [os.path.join(os.getcwd(), 'beachgray', file) for file in os.listdir('beachgray')]
MTN_PATHS_G = [os.path.join(os.getcwd(), 'mountaingray', file) for file in os.listdir('mountaingray')]
CAVE_PATHS_G = [os.path.join(os.getcwd(), 'cavegray', file) for file in os.listdir('cavegray')]
CITY_PATHS_G = [os.path.join(os.getcwd(), 'citygray', file) for file in os.listdir('citygray')]

CLASSES = [PLAIN_PATHS_G, BEACH_PATHS_G, MTN_PATHS_G, CAVE_PATHS_G, CITY_PATHS_G]

def show(file):
    '''
    Helper function that displays filename or ndarray.
    '''
    if type(file) == str: # Open filename
        image = cv2.imread(file)
    else: # Already an ndarray
        image = file

    cv2.imshow('File',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_in(filename):
    '''
    Returns tuple of BGR, YUV np-arrays of the image.
    '''
    bgr = cv2.imread(filename)
    yuv = cv2.cvtColor(bgr,cv2.COLOR_BGR2YUV)

    return bgr, yuv, cv2.split(yuv)[0]


def save_grayscale(filename, path):
    '''
    Caution: ONLY RUN THIS FUNCTION ONCE.
    Saves grayscale versions to use in keras preprocessing.
    Filename: og file name.
    Path: Path we're saving to
    '''
    graysc = read_in(filename)[2]
    cv2.imwrite(path, graysc)


def resizeAndPad(img, size=(224, 224), padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def keras_preprocess(filepath):
    '''
    Does preprocessing on the image (given by filepath) as seen in
    common pretrained NNs. Takes care of reshaping dimensions for 
    each image, etc.
    '''
    img = image.load_img(filepath, target_size=(40, 40))
    x = image.img_to_array(img)

    x = x[:,:,0]
    # assert np.shape(x) == (30, 30)

    return x # input to the NN


def load_data():
    '''
    Builds the train_x, train_y, test_x, test_y tensors to feed into NN.
    Add support for cross-validation later!
    '''
    x_train, y_train = [], []
    x_test, y_test = [], []

    for i in range(780): # Add to training subsets
        for j, PATHS in enumerate(CLASSES):
            x_train.append(keras_preprocess(PATHS[i]))
            encoding = [0, 0, 0, 0, 0]
            encoding[j] = 1
            y_train.append(encoding)

    for i in range(780, 1170): # Add to test subsets
        for j, PATHS in enumerate(CLASSES):
            x_test.append(keras_preprocess(PATHS[i]))
            encoding = [0, 0, 0, 0, 0]
            encoding[j] = 1
            y_test.append(encoding)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))



if __name__ == '__main__':
    # Save grayscale versions of images
    # for file in os.listdir('beach'):
    #     img_path = os.path.join(os.getcwd(), 'beach', file)
    #     path = os.path.join(os.getcwd(), file[:-5] + 'gray' + '.jpeg')
    #     save_grayscale(img_path, path)

    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_test, y_test = np.array(x_test), np.array(y_test)

    # print(np.shape(x_train[0]), np.shape(y_train[0]))

    # CHANGE EVERYTHING TO 224 x 224... lol
    # for file in os.listdir('cave'):
    #     img_path = os.path.join(os.getcwd(), 'cave', file)

    #     img = cv2.imread(img_path)
    #     img = resizeAndPad(img)

    #     cv2.imwrite(file, img)
    pass
