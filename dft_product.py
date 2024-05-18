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
from show_fft import show_fft
from normalization import normalization


def dft_product(img, kernel, filter_type):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    img = np.array(img, dtype='uint8')

    kernel_resized = np.pad(kernel, int((img.shape[0]-kernel.shape[0])/2), pad_with, padder=0)

    fft_img = np.fft.fft2(img)
    fft_shift_img = np.fft.fftshift(fft_img)
    show_fft(fft_shift_img, filter_type+"_fft_shift_img")

    fft_kernel = np.fft.fft2(kernel_resized)
    fft_shift_kernel = np.fft.fftshift(fft_kernel)
    show_fft(fft_shift_kernel, filter_type+"_fft_shift_kernel")

    fft_product_result = fft_shift_img * fft_shift_kernel
    show_fft(fft_product_result, filter_type+"_fft_product_result")

    ifftshift_product_result = np.fft.ifftshift(fft_product_result)
    ifft_product_result = np.fft.ifft2(ifftshift_product_result)
    ifft_product_result = np.real(ifft_product_result)
    ifft_product_result = normalization(ifft_product_result)
    ifft_product_result = np.uint8(ifft_product_result)
    final_result = np.fft.fftshift(ifft_product_result)   # 矯正奇怪的錯位

    return final_result


if __name__ == "__main__":
    img = cv2.imread("./photos/profile_photo_501.jpg", 0)

    # Input High-pass kernel
    high_pass_kernel = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]], dtype='int8')
    high_pass_result = dft_product(img, high_pass_kernel, "high_pass")
    high_pass_result = np.where(high_pass_result > 160, high_pass_result, 0)   # 邊緣 Threshold 設定 160
    cv2.namedWindow("high_pass_result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("high_pass_result", 500, 500)
    cv2.imshow("high_pass_result", high_pass_result)

    # Input Low-pass kernel
    kernel_size = 15
    low_pass_kernel = np.ones((kernel_size, kernel_size), np.float64) / kernel_size ** 2
    low_pass_result = dft_product(img, low_pass_kernel, "low_pass")
    cv2.namedWindow("low_pass_result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("low_pass_result", 500, 500)
    cv2.imshow("low_pass_result", low_pass_result)

    cv2.waitKey(0)
