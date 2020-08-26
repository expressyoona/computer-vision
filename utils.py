from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np

def filter(source, kernel):
    w, h = source.shape
    result = source.copy()
    for x in range(1, w-1):
        for y in range(1, h-1):
            result[x][y] = source[x-1][y+1]*kernel[0][0] + source[x][y+1]*kernel[0][1]\
                        + source[x+1][y+1]*kernel[0][2] + source[x-1][y]*kernel[1][0]\
                        + source[x][y]*kernel[1][1] + source[x+1][y]*kernel[1][2]\
                        + source[x-1][y-1]*kernel[2][0] + source[x][y-1]*kernel[2][1]\
                        + source[x+1][y-1]*kernel[2][2]
    return result[w-1:h-1]

def histogram(image):
    color = ('b', 'g', 'r')
    result = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        result.append((hist, col))
    return result

def change_hue(image):
    pass

def quatization(image, clusters):
    h, w, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((h*w, 3))
    clt = MiniBatchKMeans(n_clusters=clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return quant
    # return np.hstack([image, quant])


def blur(image, size):
    return cv2.blur(image, (size, size))

# Degree
def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result