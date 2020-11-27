
from skimage.feature import hog
import numpy as np
import time

def hogtoarray(image_array):
    hog_features_data = []
    start_time = time.time()
    print("\nHOG in progress >>> ")
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
    end_time = time.time()
    print("HOG time : ", end_time - start_time, " seconds. ")
    print(">>> Done HOG\n")
    return np.float32(hog_features)