# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
 
# load the image and convert it to a floating point data type
#image = img_as_float(io.imread(args["image"]))
image = cv2.imread('doggo.png')
image_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
for row in range(image_yuv.shape[0]):
    for col in range(image_yuv.shape[1]):
        image_yuv[row][col][1] = 0
        image_yuv[row][col][2] = 0
image_grey = image_yuv
print image_grey
print image_grey.shape

cv2.imshow('bgr',image_grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
segments = slic(img_as_float(image_grey), n_segments = 100, sigma = 5)
print "SEGMENTS"
print segments 
""" 
# loop over the number of segments
for numSegments in (100, 200, 300):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments = numSegments, sigma = 5)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
#ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
"""

fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)), segments))
#ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")

# show the plots
plt.show()
