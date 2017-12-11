# cnc
Contains two implementations of automatic image colorization.

Implementation 1:
Given a training image (trained on pixels), we train a all-vs-one linear classifier
on the U and V color channels for individual pixels. Then, we predict the test image's
U and V color channels by feeding in the Y color channel one pixel at a time.

Implementation 2:
We colorize an image given a semantic prior. The prior, or image class, is obtained
by first training a NN classifier on scene recognition that outputs a probability vector
for the different scenes, using cross-entropy loss as the error function. We use 3-fold cross 
validation to avoid overfitting the model.
Then, given inputs of the image and the probability vector, we train an NN to output colorings
of the image in the U and V color channels, using RMSE as the error function.

Image classes: beach, city, lowlands, cave, glacier