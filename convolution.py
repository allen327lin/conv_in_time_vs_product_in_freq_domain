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
from scipy.signal import convolve2d
from utils import normalization, show_img

def convolution(img, kernel):
    conv_result = convolve2d(img, kernel, 'same')
    conv_result = normalization(conv_result)
    conv_result = np.uint8(conv_result)

    return conv_result


if __name__ == "__main__":
    img = cv2.imread("./photos/profile_photo_501.jpg", 0)

    # Input High-pass kernel
    high_pass_kernel = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]], dtype='int8')
    high_pass_conv_result = convolution(img, high_pass_kernel)
    high_pass_conv_result = np.where(high_pass_conv_result > 160, high_pass_conv_result, 0)
    show_img("high_pass_conv_result", high_pass_conv_result)

    # Input Low-pass kernel
    kernel_size = 15
    low_pass_kernel = np.ones((kernel_size, kernel_size), np.float64) / kernel_size ** 2
    low_pass_conv_result = convolution(img, low_pass_kernel)
    show_img("low_pass_conv_result", low_pass_conv_result)

    cv2.waitKey(0)