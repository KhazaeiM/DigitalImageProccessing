import cv2
import numpy as np
import matplotlib.pyplot as plt


def stretch(intensity, intensity_min, intensity_max, mx, mn):
    return (((intensity-intensity_min)/(intensity_max-intensity_min))*(mx-mn)) + mn


def shrink(intensity, intensity_min, intensity_max, shrink_max, shrink_min):
    return ((shrink_max - shrink_min)/(intensity_max - intensity_min))*(intensity - intensity_min) + shrink_min


def slide(intensity, offset):
    return (intensity + offset)


image = cv2.imread('image2.tif', 0)
cv2.imshow('Original Image', image)
cv2.waitKey()
# MAKE ARRAY FOR PLOTTING
height, width = image.shape
st_image = np.zeros((height, width), 'uint8')
sh_image = np.zeros((height, width), 'uint8')
sd_image = np.zeros((height, width), 'uint8')
mx = 255
mn = 0

# STRETCHING WITH STRETCH FUNCTION ON WHOLE IMAGE
for i in range(height):
    for j in range(width):
        st_image[i, j] = stretch(image[i, j], image.min(), image.max(), mx, mn)
cv2.imshow('Stretched Histogram', st_image)
cv2.waitKey()

# STRETCHING WITH STRETCH FUNCTION ON 2 PARTS OF IMAGE

# for i in range(height):
#     for j in range(width):
#         if image[i, j] <= mid:
#             st_image[i, j] = stretch(image[i, j], image.min(), image.max(), 127, mn)
#         else:
#             st_image[i, j] = stretch(image[i, j], image.min(), image.max(), mx, 128 )
# cv2.imshow('Stretched_2 Histogram', st_image)
# cv2.waitKey()

# SHRINKING WITH SHRINK FUNCTION
shrink_min = 100
shrink_max = 200
for i in range(height):
    for j in range(width):
        sh_image[i, j] = shrink(image[i, j], image.min(), image.max(), shrink_max, shrink_min)
cv2.imshow('Shrunk Histogram', sh_image)
cv2.waitKey()

# SLIDING WITH SLID FUNCTION BASED ON HISTOGRAM OF IMAGE IN LEFT OR RIGHT TO DETERMINE OFFSET'S SIGN
upper = 0
lower = 0
for i in range(height):
    for j in range(width):
        if image[i, j].mean() >= 127:
            upper += 1
        else:
            lower += 1
offset = round(image.max() - image.min())
# IMAGE IS SO LIGHT HENCE OFFSET WILL BE NEGATIVE
if upper >= lower:
    offset = -offset
for i in range(height):
    for j in range(width):
        sd_image[i, j] = slide(image[i, j], offset)
cv2.imshow('Slided Histogram', sd_image)
cv2.waitKey()

# FOR PLOTTING ALL HISTOGRAMS
data = [image.flatten(), st_image.flatten(), sh_image.flatten(), sd_image.flatten()]
x_axes = 'Intensity Value'
y_axes = 'Count'
titles = ['Original Histogram', 'Stretched Histogram', 'Shrunk Histogram in [100,200]', 'Slided Histogram']

fig, axe = plt.subplots(2, 2)
axe = axe.ravel()
for idx, ax in enumerate(axe):
    ax.hist(data[idx], bins=255, alpha=0.75, histtype='bar', ec='black')
    ax.set_title(titles[idx])
    ax.set_xlabel(x_axes)
    ax.set_ylabel(y_axes)
    ax.set_xlim([0, 257])
plt.tight_layout()
plt.show()