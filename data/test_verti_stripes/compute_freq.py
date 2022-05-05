
import imageio
import cv2
import numpy as np
image = imageio.imread("../test/1spp/0.hdr")
rgb_weights = [ 0.1140, 0.2989, 0.5870]
greyscale = np.dot(image.permute(0,3,2,1)[...,:3], rgb_weights)
f_pred = np.fft.fft2(greyscale)
magnitude_spectrum_pred = 20* np.log(np.abs(np.fft.fftshift(f_pred)))
cv2.imwrite("test.hdr", magnitude_spectrum_pred)
cv2.imwrite("greyscale.hdr", greyscale)
