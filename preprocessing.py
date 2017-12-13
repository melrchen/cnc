import pylab as pl
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import sklearn
import sklearn.cluster
import os

def color_dist(c1, c2):
    '''
    Helper function that defines the distance between two pixels.

    Args:
        ci (list)

    Returns:
        dist (int)
    '''
    dist = 0
    for element in c1-c2:
        dist += abs(element)
    return dist 

#image=img.imread('doggo.png')
#imgplot = plt.imshow(image)
#plt.show()



def read_in(filename):
    '''
    Reads in image filename.

    Args:
        filename (str)

    Returns:
        bgr, yuv tuple of np.arrays
    '''

    # BGR Image
    bgr = cv2.imread(filename)
    # print("bgr size")
    # print(bgr.shape)
    #cv2.imshow('BGR',bgr)

    #YUV image
    yuv = cv2.cvtColor(bgr,cv2.COLOR_BGR2YUV)
    # print("yuv size")
    # print(yuv.shape)
    # cv2.imshow('YUV',yuv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return bgr, yuv


def discretize_colors(filename):
    '''
    Takes in filename and outputs discretized pixel color mapping.

    Args:
        filename (string)

    Returns:
        pixel_color_map (dict): Mapping of coordinate (tuple) to color (np array YUV)
    '''
    bgr, yuv = read_in(filename)
    pixel_color_map = {} # Maps tuple of pixel coordinate to [Y, U, V] vector
    pixel_color_array = np.zeros(yuv.shape)

    #KMeans to find colors
    w, h, d = original_shape = tuple(yuv.shape)
    assert d == 3
    image_array = np.reshape(yuv, (w * h, d))

    kmeans = sklearn.cluster.KMeans(n_clusters=8).fit(image_array)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # print("labels")
    # print(labels)
    # print("cluster centers")
    # print(cluster_centers)

    """
    #picture of cluster colors
    cluster_pic = []
    for cluster in cluster_centers:
        for ii in range(0, 50):
            col = []
            for jj in range(0, 50):
                col.append(cluster)
            cluster_pic.append(np.array(col))
    cluster_pic = np.array(cluster_pic)
    cluster_pic = cluster_pic.astype(np.uint8)
    #print "CLUSTER SHAPE"
    #print cluster_pic.shape
    #print cluster_pic

    bgr_clusters = cv2.cvtColor(cluster_pic,cv2.COLOR_YUV2BGR)
    cv2.imshow('clusters', bgr_clusters)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    for ii in range(0, yuv.shape[0]):
        for jj in range(0, yuv.shape[1]):
            pixel = (ii, jj)
            best = cluster_centers[0]
            best_dist = color_dist(yuv[ii][jj], cluster_centers[0])

            for color in cluster_centers:
                dist = color_dist(yuv[ii][jj], color)
                if best_dist > dist:
                    best_dist = dist
                    best = color
            pixel_color_map[pixel] = best
            pixel_color_array[ii][jj] = best

    return pixel_color_array


    # bgr_map = cv2.cvtColor(pixel_color_array.astype(np.uint8),cv2.COLOR_YUV2BGR)
    # cv2.imshow('lol', bgr_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # 0-255?

    path = os.path.join(os.getcwd(), 'city.jpeg')
    yuv = read_in(path)[1]

    print(yuv.shape)
    y, u, v = cv2.split(yuv)
    # y = yuv[:,:,0]
    # print(y)
    # cv2.imshow('y', y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(np.amin(u), np.amax(u))
    # print(np.amin(v), np.amax(v))
    # print(u[120])
    # print(u[1])
    print(u.shape)
    print(type(u[0][0]))
    cv2.imshow('u', u)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(v)
    cv2.imshow('y', v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

