import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift

img = cv2.imread('home.tif', 0)
# Fourier Tarnsform
fourier = fft2(img)
# Shift the zero-frequency component to the center of the spectrum
fourier_shift = fftshift(fourier)
# Using LOG for better visualization
mag_spec = np.abs(fourier_shift)
magnitude_spectrum = np.log(mag_spec)
phase_spectrum = np.abs(np.angle(fourier_shift))

f_ishift = ifftshift(fourier_shift)
# inverse fft to get the image back
img_back = np.abs(ifft2(f_ishift))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
axs1 = ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax1.axis('off')
axs2 = ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.title.set_text('Spectrum')
ax2.axis('off')
axs3 = ax3.imshow(phase_spectrum, cmap='gray')
ax3.title.set_text('Phase Angle')
ax3.axis('off')
axs4 = ax4.imshow(img_back, cmap='gray')
ax4.title.set_text('Inverted Image')
ax4.axis('off')
plt.show()

# Get the center pixel
rows, cols = img.shape[:2]
mid_row, mid_col = int(rows / 2), int(cols / 2)

# create a mask first, center square is 1, remaining all zeros
mask_l = np.zeros((rows, cols), np.uint8)
mask_l[mid_row-30:mid_row+30, mid_col-30:mid_col+30] = 1
# apply mask and inverse DFT
l_fshift = fourier_shift*mask_l
l_p_filter = np.abs(ifft2(l_fshift))

h_fshift = fourier_shift
h_fshift[mid_row-30:mid_row+30, mid_col-30:mid_col+30] = 0
h_p_filter = np.abs(ifft2(ifftshift(h_fshift)))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
axs1 = ax1.imshow(img, cmap='gray')
ax1.axis('off')
ax1.title.set_text('Input Image')
axs2 = ax2.imshow(l_p_filter, cmap='gray')
ax2.axis('off')
ax2.title.set_text('Lowpass Filter')
axs3 = ax3.imshow(h_p_filter, cmap='gray')
ax3.title.set_text('Highpass Filter')
ax3.axis('off')
plt.show()
