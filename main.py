import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import rotate
from utils import quatization
from utils import blur
from utils import histogram


img = cv2.imread('./images/rose.jpeg')

plt.figure('Point operator on image')

rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(231)
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.imshow(rgb_image)

s = cv2.convertScaleAbs(rgb_image, beta=20)
plt.subplot(232)
plt.title('Increase contrast')
plt.xticks([]), plt.yticks([])
plt.imshow(s)

rotated = rotate(rgb_image, 45)
plt.subplot(233)
plt.title('Rotated 45 degree')
plt.xticks([]), plt.yticks([])
plt.imshow(rotated)

quantized = quatization(rgb_image, 4)
plt.subplot(234)
plt.title('Quantized colors')
plt.xticks([]), plt.yticks([])
plt.imshow(quantized)

blurred = blur(rgb_image, 9)
plt.subplot(235)
plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.imshow(blurred)

img_yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
plt.subplot(236)
plt.title('Histogram Equalized')
plt.xticks([]), plt.yticks([])
plt.imshow(equalized)

plt.show()
"""
hist = histogram(img)
for h, c in hist:
    plt.plot(h, color=c)
    plt.xlim([0, 256])
plt.show()
"""