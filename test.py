import numpy as np 
from svms import *
import featurization as ft 
import preprocessing as pp
import cv2

def set_classifiers():
    classifiers = []
    W = ([[ -9.28925484e-05,   3.80354647e-05,   9.35573205e-05,   6.95032257e-05,
    1.40380185e-05,  -1.19112193e-04,  -1.03455593e-04,   1.94140685e-04,
    1.81839260e-05,  -1.08158250e-04,  -3.10956460e-04,   4.80032685e-05,
    7.06855315e-05, -2.17525223e-05,   8.18700648e-05,   1.91371576e-05,
   -1.33698007e-04,-3.17416191e-04,   4.60564588e-05,   1.00852289e-04,
   -4.25839813e-05,  -3.78474982e-05,  -1.13306625e-04,   2.92857503e-04,
    1.61436536e-04,  -1.16798315e-04,  -1.34358898e-04,   1.15112378e-04,
    1.89481729e-04,   1.46861037e-04]],
    [[  2.34823721e-05,  -3.20591229e-05,  -2.60287095e-05,   6.25465076e-05,
   -1.05368579e-04,  -3.28636859e-05,   2.21276109e-05,  -4.84205441e-05,
   -1.70482028e-05,   1.33450454e-05,   5.04203250e-06,  -4.97765953e-05,
    2.22889683e-05,   1.67048154e-05,  -4.45748278e-05,  -5.33462673e-07,
    2.51624999e-06,  -5.74329740e-05,  -1.08995041e-05,  -4.07143095e-05,
    1.52591862e-05,  -1.88415204e-05,  -6.35787747e-06,  -2.13193688e-05,
   -1.31610002e-04,   8.89601606e-05,   8.64202014e-05,  -6.33754687e-05,
   -7.43911344e-05,  -8.42845692e-05]],
    [[2.71153920e-05,  -1.66193697e-04,  -3.76963252e-05,   1.14675363e-05,
    5.03525765e-05,   1.14228667e-05,   3.93423413e-05,   5.07321621e-05,
   -8.38760842e-06,   2.18307562e-05,  -1.65964773e-05,  -1.01744726e-04,
    1.19871059e-04,   2.09189461e-05,   2.83519784e-05,  -1.09485615e-05,
    1.45204850e-05,  -6.93405047e-06,  -4.22469874e-05,   1.06890798e-04,
   -1.03168620e-05,   3.45634983e-05,  -3.54114691e-05,  -6.77377923e-05,
   -4.08751469e-05,  -9.33014513e-06,  -2.62725700e-06,  -2.58697354e-05,
    5.94514081e-05,   1.61716171e-06]],
    [[  2.92054433e-05,  -1.69577262e-04,   4.60488950e-05,   1.09640101e-04,
    2.09815240e-04,  -7.45977651e-05,   1.53537692e-05,   3.94691166e-05,
    2.37932202e-05,   2.32893516e-05,   5.51935089e-06,  -4.28380672e-06,
    3.49109132e-05,   6.49098704e-05,  -5.93971733e-05,  -6.47042137e-05,
    2.25243175e-05,  -1.38338702e-04,  -4.37748129e-05,  -5.60000519e-05,
   -5.99706428e-06,  -5.41754107e-05,   6.24685113e-05,  -4.66783795e-05,
   -2.91150617e-05,   1.72071750e-04,  -2.93030258e-05,   2.91545148e-05,
    8.74318969e-05,   2.26282890e-05]],
    [[  3.32866100e-05,  -7.13701325e-05,   2.63286660e-06,   5.77425030e-05,
    3.09622580e-05,  -7.46128115e-05,  -1.37210188e-04,   1.53805672e-04,
   -7.59699009e-05,   5.28202962e-06,  -4.87801970e-05,  -4.48301989e-05,
    1.20534354e-04,   6.52176501e-06,   4.30838556e-05,  -6.97405030e-05,
   -8.29342022e-06,   6.58335166e-05,  -3.73312653e-07,   8.92161182e-05,
   -1.93723330e-05,   3.35106523e-05,   3.09981523e-05,   2.88906334e-06,
   -1.61189320e-05,  -4.71533548e-05,  -8.01853037e-05,  -7.07390873e-05,
   -5.28178438e-05,  -1.51959732e-05]],
   [[ -5.66654465e-05,   3.65940425e-05,  -1.60482702e-04,  -2.18963431e-04,
    2.08668203e-06,  -1.45956982e-04,   2.80929611e-05,  -6.93544505e-06,
    1.81267391e-05,   1.06648536e-04,   2.26318471e-05,  -2.25952132e-05,
   -1.17471727e-05,  -1.02974655e-04,   1.80356054e-05,  -9.76558404e-06,
    1.68520171e-04,  -2.38156996e-05,  -7.00341858e-06,  -5.21125935e-05,
   -1.30708748e-04,  -7.80128337e-05,   2.19567171e-04,  -3.32245940e-06,
    2.43887812e-05,   1.38346316e-04,   8.02997046e-05,   1.04755081e-04,
   -6.16318872e-05,   5.75143893e-05]],
   [[  2.92701882e-05,  -6.42270367e-05,   1.22693487e-05,   6.31066587e-05,
    5.87502357e-05,  -2.04763620e-04,   3.17585504e-06,   8.31649038e-05,
    9.19902083e-06,   1.34987874e-04,   1.20515505e-04,  -4.87482889e-05,
    9.16257968e-06,  -1.98269073e-05,  -4.93144031e-05,   1.13071199e-05,
    3.48882029e-05,   1.87199808e-05,  -3.12493714e-05,   1.01581680e-04,
   -6.20153398e-05,  -4.15236396e-05,  -5.22395366e-05,   4.44179873e-05,
    1.60312186e-05,   6.55226320e-05,  -3.28017434e-05,   3.80365198e-05,
    5.21361051e-05,   5.53720123e-05]],
    [[  3.65917081e-05,   7.38999903e-05,  -6.98845144e-05,   5.01639826e-05,
    7.04474594e-05,  -4.04466109e-05,  -9.35656462e-05,  -6.55323054e-05,
   -1.01867801e-04,   2.25141592e-05,  -1.22396581e-04,  -7.11237380e-05,
   -6.19148743e-05,   1.35050653e-05,   1.04068478e-05,   2.42508496e-04,
   -1.03882719e-04,   1.01197699e-04,  -6.06251842e-05,  -7.33900177e-05,
   -2.95428028e-04,   2.20039071e-04,   4.81466524e-05,   1.87843647e-04,
    2.52658588e-04,   1.03678897e-04,  -6.74932582e-05,  -6.30178066e-05,
   -6.32478509e-05,   1.11252829e-04]]
    )
    W0 = [[-0.23187706], [-0.69451225], [-0.67720976], [-0.6669709], [-0.5932561], [-0.4396003], [-0.64611823], [-0.50983935]]
    colors = [(127.96087059240629, 128.11453349868117), 
    (114.00805214724052, 143.67937116564462), (110.40860288586626, 147.7852697804833), 
    (99.347312204519213, 162.72868149185831), (101.01220729366759, 164.80268714011567), 
    (109.7850725639266, 158.41856715042667), (113.63500223513654, 144.69378632096604), 
    (121.2834055971621, 143.80469057942497)]

    for i in range(len(colors)):
        clf = svm.LinearSVC(random_state = 0, multi_class='ovr') # ovr: one vs. rest
        clf.coef_ = np.array(W[i])
        clf.intercept_ = np.array(W0[i])
        classifiers.append(clf)

    return colors, classifiers




def test(filename, trainingfile):
    '''
    Tests our algorithm on an image file.

    Args:
        filename (string)

    Returns:
        Score (float): Distance between actual and predicted
    '''
    # Featurize image
    print('==========Featurizing image==========')
    feature_vec = ft.featurize(filename)

    # Get SVM classifiers
    colors, classifiers = SVM_classification(trainingfile)
    # print(colors)
    # colors, classifiers = set_classifiers()


    print("=========Choosing best color for each pixel===========")
    predicted = pp.read_in(filename)[1] 
    prelim = []
    for pixel in feature_vec:
        # Use all the classifiers
        scores, confidence = [], []
        pixel = pixel.reshape(pixel.shape[0], 1).T
        for classifier in classifiers:
            scores.append(classifier.predict(pixel)[0])
            # print(classifier.predict(pixel), classifier.decision_function(pixel))
            confidence.append(classifier.decision_function(pixel)[0])

        # Compare performance by finding the 1-score with the greatest 
        # confidence. If there are no 1's, find the 0-score with the lowest
        # confidence.
        if 1 in scores:
            indices1 = [i for i in range(len(scores)) if scores[i] == 1]
            maxconf, index = 0, -1
            
            for i in indices1:
                if confidence[i] > maxconf:
                    maxconf, index = confidence[i], i
        else:
            minconf, index = float('inf'), -1
            for i, conf in enumerate(confidence):
                if conf < minconf:
                    minconf, index = conf, i

        color = colors[index]
        prelim.append(color)

    print('==========BUILDING OUTPUT MATRIX===========')
    # Add colors to predicted matrix
    k = 0 # index of prelim
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            predicted[i][j][1] = prelim[k][0]
            predicted[i][j][2] = prelim[k][1]
            k += 1

    # Convert to BGR
    bgr = cv2.cvtColor(predicted,cv2.COLOR_YUV2BGR)
    cv2.imshow('BGR',bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    test('doggo2.png', 'doggo.png')
    # print(set_classifiers())