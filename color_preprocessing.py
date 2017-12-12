import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential, model_from_json
from keras.layers import Conv2D, BatchNormalization as BN
import cv2
import scipy as sp
import numpy as np
import os

base_model = VGG19(weights='imagenet')

# Feature extraction
feature_model = Model(inputs=base_model.input, outputs=
    [base_model.get_layer('block1_conv2').output,
    base_model.get_layer('block3_conv4').output,
    base_model.get_layer('block5_conv4').output])


def get_hypercolumn(filepath):
    '''
    Retrieves hypercolumn (without upsampling) from base model
    VGG19 given image file path. 
    '''
    img_path = filepath

    # Image preprocessing
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x.astype('float32')
    x /= 255

    hypercolumns = feature_model.predict(x) # Get hypercolumns from model

    return hypercolumns


def upsample_hypercolumn(filepath):
    '''
    Based on the hypercolumn, upsamples everything to a suitable
    size
    '''
    hypercolumns = get_hypercolumn(filepath) # List of feature tensors
    
    upsampled = []
    # Process hypercolumns to be the right shape
    for i, hypercolumn in enumerate(hypercolumns):
        convmaps = np.reshape(hypercolumn, (hypercolumn.shape[-1], 
            hypercolumn.shape[1], hypercolumn.shape[2]))

        for i, convmap in enumerate(convmaps): # Add upscaled feature maps to upsampled
            if i % 8 == 0:
                upscaled = sp.misc.imresize(convmap, size=(224, 224),
                                        mode="F", interp='bilinear')
                upsampled.append(upscaled)
    
    upsampled = np.array(upsampled)
    upsampled = np.reshape(upsampled, (upsampled.shape[1], upsampled.shape[2], 
        upsampled.shape[0]))
    
    return upsampled


def UVchannels(filepath):
    '''
    Given file path, extracts concatenated U, V channels
    '''
    img = cv2.imread(filepath)
    yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    __, U, V = cv2.split(yuv)

    U = np.reshape(U, (224, 224, 1))
    V = np.reshape(V, (224, 224, 1))

    x = np.concatenate((U/128, V/128), -1) # Converts to -1, 1
    return x


def load_data():
    '''
    Loads just the beach pics for now

    Returns:
        Tuple of tuples (x train, y train), (x test, y test)
        where x_train is a bunch of upsampled hypercolumns of images
        and y_train is the concatenation of U and V color channels
    '''
    x_train, y_train, x_test, y_test = [], [], [], []

    PATHS = [os.path.join(os.getcwd(), 'city', file) for file in os.listdir('city')]
    GRAYPATHS = [os.path.join(os.getcwd(), 'citygray', file) for file in os.listdir('citygray')]

    print('Aggregating training data')
    for i in range(400): # Add grayscale images to x_train
        x_train.append(upsample_hypercolumn(GRAYPATHS[i]))
        y_train.append(UVchannels(PATHS[i]))
        # print(i)

    print('Aggregating validation data')
    for i in range(400, 500): # Add grayscale images to x_test
        x_test.append(upsample_hypercolumn(GRAYPATHS[i]))
        y_test.append(UVchannels(PATHS[i]))

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


if __name__ == '__main__':
    # wtf is going on with the loss?
    path = os.path.join(os.getcwd(), 'plaingray.jpeg')

    img = cv2.imread(path)

    # print('Y channel: ',img[100]/255)
    # get_hypercolumn(path)
    # upsample_hypercolumn(path)

    # UVchannels(path)
