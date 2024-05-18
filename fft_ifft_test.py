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

img = cv2.imread("./photos/profile_photo_255.jpg", 0)

# kernel = np.array([[ 0,-1, 0],
#                              [-1, 4,-1],
#                              [ 0,-1, 0]], dtype='int8')
#
# def pad_with(vector, pad_width, iaxis, kwargs):
#     pad_value = kwargs.get('padder', 10)
#     vector[:pad_width[0]] = pad_value
#     vector[-pad_width[1]:] = pad_value
#
# img = np.pad(kernel, 126, pad_with, padder=0)

# fft

fft_img = np.fft.fft2( img )
fft_shift_img = np.fft.fftshift( fft_img )    #將0頻率分量移動到影象的中心

fft_shift_img_1 = 1 * np.log( np.abs( fft_shift_img ) + 1 )
fft_shift_img_1 = fft_shift_img_1 / fft_shift_img_1.max( ) * 255.0
fft_shift_img_1 = np.uint8( fft_shift_img_1 )

cv2.namedWindow("fft_shift_img_1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("fft_shift_img_1", 750, 750)
cv2.imshow("fft_shift_img_1", fft_shift_img_1)

fft_shift_img_20 = 20 * np.log( np.abs( fft_shift_img ) + 1 )
fft_shift_img_20 = fft_shift_img_20 / fft_shift_img_20.max( ) * 255.0
fft_shift_img_20 = np.uint8( fft_shift_img_20 )

cv2.namedWindow("fft_shift_img_20", cv2.WINDOW_NORMAL)
cv2.resizeWindow("fft_shift_img_20", 750, 750)
cv2.imshow("fft_shift_img_20", fft_shift_img_20)

# ifft

ifftshift_img = np.fft.ifftshift(fft_shift_img)
ifft_img = np.fft.ifft2(ifftshift_img)
ifft_img = np.real(ifft_img)
ifft_img = np.uint8( ifft_img )

cv2.namedWindow("ifft_img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ifft_img", 750, 750)
cv2.imshow("ifft_img", ifft_img)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("img", 750, 750)
cv2.imshow("img", img)

cv2.waitKey(0)
