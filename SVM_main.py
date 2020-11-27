import os

from sklearn import svm

import include.pca_reduction
import include.lda_reduction
import include.test_accuracy
import include.hogtoarray

import time
import numpy as np
import gzip


def main():

    # train_image, train_label = utils.mnist_reader.load_mnist('data/fashion', kind='train')
    # test_image, test_label = utils.mnist_reader.load_mnist('data/fashion', kind='t10k')
    train_image, train_label = load_mnist('fashion-mnist/data/fashion', kind='train')
    test_image, test_label = load_mnist('fashion-mnist/data/fashion', kind='t10k')
    print("\n\n-------------------------\nOn data processing: \n-------------------------")
    hog_train = include.hogtoarray.hogtoarray(train_image)
    hog_test = include.hogtoarray.hogtoarray(test_image)
    pca_train, pca_test = include.pca_reduction.pca_reduction(train_image, test_image)
    lda_train, lda_test = include.lda_reduction.lda_reduction(train_image, train_label, test_image)
    #
    # hog_train =include.hog_for_array(pca_train)
    # hog_test = include.hog_for_array(pca_train)

    print("\n\n-------------------------\nOn testing set: \n-------------------------")

    start_time = time.time()
    print("\nSVM poly in progress >>> ")
    svm_clf = svm.SVC(kernel='poly', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    poly_predict = svm_clf.predict(pca_test)
    poly_accuracy_pca = include.test_accuracy.test_accuracy(poly_predict, test_label)
    print("SVM poly accuracy on pca:", poly_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    poly_predict = svm_clf.predict(lda_test)
    poly_accuracy_lda = include.test_accuracy.test_accuracy(poly_predict, test_label)
    print("SVM poly accuracy on lda:", poly_accuracy_lda)
    end_time = time.time()
    print("SVM poly time : ", end_time - start_time, " seconds. ")
    print(">>> Done SVM poly\n")

    start_time_poly = time.time()
    print("\nSVM rbf in progress >>> ")
    svm_clf = svm.SVC(kernel='rbf', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    rbf_predict = svm_clf.predict(pca_test)
    rbf_accuracy_pca = include.test_accuracy.test_accuracy(rbf_predict, test_label)
    print("SVM rbf accuracy on pca:", rbf_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    rbf_predict = svm_clf.predict(lda_test)
    rbf_accuracy_lda = include.test_accuracy.test_accuracy(rbf_predict, test_label)
    print("SVM rbf accuracy on lda:", rbf_accuracy_lda)
    end_time_poly = time.time()
    print("SVM rbf time : ", end_time_poly - start_time_poly, " seconds. ")
    print(">>> Done SVM rbf\n")

    print("\n\n-------------------------\nOn training set: \n-------------------------")
    start_time = time.time()
    print("\nSVM poly in progress >>> ")
    svm_clf = svm.SVC(kernel='poly', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    poly_predict = svm_clf.predict(pca_train)
    poly_accuracy_pca = include.test_accuracy.test_accuracy(poly_predict, train_label)
    print("SVM poly accuracy on pca:", poly_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    poly_predict = svm_clf.predict(lda_train)
    poly_accuracy_lda = include.test_accuracy.test_accuracy(poly_predict, train_label)
    print("SVM poly accuracy on lda:", poly_accuracy_lda)
    end_time = time.time()
    print("SVM poly time : ", end_time - start_time, " seconds. ")
    print(">>> Done SVM poly\n")

    start_time_poly = time.time()
    print("\nSVM rbf in progress >>> ")
    svm_clf = svm.SVC(kernel='rbf', gamma='scale')
    svm_clf.fit(pca_train, train_label)
    rbf_predict = svm_clf.predict(pca_train)
    rbf_accuracy_pca = include.test_accuracy.test_accuracy(rbf_predict, train_label)
    print("SVM rbf accuracy on pca:", rbf_accuracy_pca)

    svm_clf.fit(lda_train, train_label)
    rbf_predict = svm_clf.predict(lda_train)
    rbf_accuracy_lda = include.test_accuracy.test_accuracy(rbf_predict, train_label)
    print("SVM rbf accuracy on lda:", rbf_accuracy_lda)
    end_time_poly = time.time()
    print("SVM rbf time : ", end_time_poly - start_time_poly, " seconds. ")
    print(">>> Done SVM rbf\n")

# def lda_reduction(input_train, input_train_label, input_test, components_number=None):
#     start_time = time.time()
#     print("\nLDA in progress >>> ")
#     lda = LDA(n_components=components_number)
#     lda.fit(input_train, input_train_label)
#     lda_train = lda.transform(input_train)
#     lda_test = lda.transform(input_test)
#     end_time = time.time()
#     print("LDA time : ", end_time - start_time, " seconds. ")
#     print(">>> Done LDA\n")
#     return lda_train, lda_test
#
#
#
# def pca_reduction(input_train, input_test, pca_target_dim=30):
#     start_time = time.time()
#     print("\nPCA in progress >>> ")
#     if pca_target_dim:
#         pca = PCA(n_components=pca_target_dim)
#         print("PCA target dimension chosen as: ", pca.n_components)
#     else:
#         pca = PCA()
#         print("PCA target dimension selected as auto")
#     pca.fit(input_train)
#     pca_train = pca.transform(input_train)
#     pca_test = pca.transform(input_test)
#     end_time = time.time()
#     print("PCA time : ", end_time - start_time, " seconds. ")
#     print(">>> Done PCA\n")
#     return pca_train, pca_test
#
#
#
#
# def test_accuracy(predicted_cat, labeled_cat):
#     predicted_cat_arr = np.array(predicted_cat)
#     labeled_cat_arr = np.array(labeled_cat)
#     diff = predicted_cat_arr - labeled_cat_arr
#     false_count = np.count_nonzero(diff)
#     accuracy = 1 - false_count / predicted_cat_arr.shape[0]
#     return accuracy

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



if __name__ == '__main__':
    main()
