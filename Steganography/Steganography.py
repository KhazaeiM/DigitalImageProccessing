import cv2
import numpy as np

# Read Images in Grayscale Mode
image = np.asarray(cv2.imread('Leopards.jpg', 0))
ai, bi = image.shape
cv2.imshow("Original Image", image)
cv2.waitKey()

target = np.asarray(cv2.imread('Cubic.jpg', 0))
at, bt = target.shape

t_array = np.zeros((at, 3*bt), "uint8")
i_array = image
arr = np.zeros((ai, bi), 'uint8')
# Decode Target Image
for i in range(at):
    for j in range(bt):
        t_array[i, j*3] = (target[i, j] & 192) >> 6
        t_array[i, j*3+1] = (target[i, j] & 48) >> 4
        t_array[i, j*3+2] = (target[i, j] & 12) >> 2

for i in range(at):
    for j in range(bt*3):
        i_array[i, j] = (image[i, j] & ~3) + t_array[i, j]

cv2.imshow("Combined Image", i_array)
cv2.waitKey()

# Encode
for i in range(ai):
    for j in range(int(bi/3)):
        arr[i, j] = ((i_array[i, 3*j]&3)<<6) + ((i_array[i, 3*j+1]&3)<<4) + ((i_array[i, 3*j+2]&3)<<2) + 3
arr = arr[:at, :bt]

cv2.imshow("Secret Image", arr)
cv2.waitKey()
