import cv2
import numpy as np
import matplotlib.pyplot as plt

red, green, blue = np.zeros((256,), dtype=int), np.zeros((256,), dtype=int), np.zeros((256,), dtype=int)

IMAGE_PATH = './images/rose.jpeg'

img = cv2.imread(IMAGE_PATH)
w, h, _ = img.shape
for x in range(w):
    for y in range(h):
        b, g, r = img[x][y]
        # print(b,g,r)
        red[r] += 1
        blue[b] += 1
        green[g] += 1

x = np.arange(256)

plt.plot(x, blue, label="Blue", color="blue")

plt.plot(x, green, label="Green", color="green")


plt.plot(x, red, label="Red", color="red")


plt.xlabel('Subjects')
plt.ylabel('Marks')

plt.title('Histogram')

plt.legend()

plt.show()