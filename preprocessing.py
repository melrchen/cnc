import pylab as pl
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import sklearn
import sklearn.cluster

def color_dist(c1, c2):
    dist = 0
    for element in c1-c2:
        dist += abs(element)
    return dist 

#image=img.imread('doggo.png')
#imgplot = plt.imshow(image)
#plt.show()



#BGR image
bgr = cv2.imread('doggo.png')
print "bgr size"
print bgr.shape
#cv2.imshow('BGR',bgr)

#YUV image
yuv = cv2.cvtColor(bgr,cv2.COLOR_BGR2YUV)
print "yuv size"
print yuv.shape
#cv2.imshow('YUV',yuv)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

pixel_color_map = {}
pixel_color_array = np.zeros(yuv.shape)

#KMeans to find colors
w, h, d = original_shape = tuple(yuv.shape)
assert d == 3
image_array = np.reshape(yuv, (w * h, d))

kmeans = sklearn.cluster.KMeans(n_clusters=8).fit(image_array)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
#cluster_set = set(cluster_centers)
print "labels"
print labels
print "cluster centers"
print cluster_centers

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
        #if yuv[ii][jj] in cluster_set:
        #    pixel_color_map[pixel] = yuv[ii][jj]
        #    pixel_color_array[ii][jj] = yuv[ii][jj]
        #else :
        best = cluster_centers[0]
        best_dist = color_dist(yuv[ii][jj], cluster_centers[0])
        for color in cluster_centers:
            dist = color_dist(yuv[ii][jj], color)
            if best_dist > dist:
                best_dist = dist
                best = color
        pixel_color_map[pixel] = best
        pixel_color_array[ii][jj] = best

bgr_map = cv2.cvtColor(pixel_color_array.astype(np.uint8),cv2.COLOR_YUV2BGR)
cv2.imshow('lol', bgr_map)
cv2.waitKey(0)
cv2.destroyAllWindows()


                
