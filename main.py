"""
MIT License

Copyright (c) 2024 allen327lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2
from resize_kernel import resize_kernel
from show_fft import show_fft

original_img = cv2.imread("./photos/profile_photo_255.jpg", 0)

# -----

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

img = original_img.copy()
img = np.array(img, dtype='uint8')
high_pass_kernel = np.array([[ 0,-1, 0],
                                    [-1, 4,-1],
                                    [ 0,-1, 0]], dtype='int8')
high_pass_kernel_resized = np.pad(high_pass_kernel, 126, pad_with, padder=0)

fft_img = np.fft.fft2(img)
fft_shift_img = np.fft.fftshift(fft_img)
show_fft(fft_shift_img, "fft_shift_img")

fft_kernel = np.fft.fft2(high_pass_kernel_resized)
fft_shift_kernel = np.fft.fftshift(fft_kernel)
show_fft(fft_shift_kernel, "fft_shift_kernel")

fft_product_result = fft_shift_img * fft_shift_kernel
show_fft(fft_product_result, "fft_product_result")

ifft_product_result = np.fft.ifftshift(fft_product_result)
ifft_product_result = np.fft.ifft2(ifft_product_result)
ifft_product_result = np.real(ifft_product_result)
ifft_product_result = np.uint8(ifft_product_result)

cv2.namedWindow("ifft_product_result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ifft_product_result", 750, 750)
cv2.imshow("ifft_product_result", ifft_product_result)
cv2.waitKey(0)

# -----




low_pass_kernel = np.ones((3,3), np.uint8)


# cv2.namedWindow("profile_photo", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("profile_photo", 750, 750)
# cv2.imshow("profile_photo", img)
# cv2.waitKey(0)