# Linfeng Li
# Xing Qian
# CS 412
# University of Illinois at Chicago
# 11/20/2020

import gzip
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



def knn_from_library(k, x_train, y_train):
    x_train = hog_for_array(x_train)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    return model




def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def hog_for_array(image_array):
    hog_features_data = []
    for img in image_array:
        img = img.reshape(28, 28)
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(7, 7),
                 cells_per_block=(4, 4))
        hog_features_data.append(fd)
        # plt.hist(fd)
        # plt.show()
    hog_features = np.array(hog_features_data, 'float64')
    return np.float32(hog_features)


def predict(model, x_test):
    x_test = hog_for_array(x_test)
    return model.predict(x_test)


def main():
    plt.gray()
    x_train, y_train = load_mnist('fashion-mnist/data/fashion', kind='train')
    x_test, y_test = load_mnist('fashion-mnist/data/fashion', kind='t10k')
    feature = hog_for_array(x_train[0:10])
    # label = y_train[0:10]

    model = knn_from_library(3, x_train, y_train)
    y_pred = predict(model, x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # print("prediction is", model.predict(hog_for_array(x_test[0:1])))
    # print("true label is", y_test[0])



if __name__ == '__main__':
    main()