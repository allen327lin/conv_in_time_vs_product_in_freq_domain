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
from dft_product import dft_product
from convolution import convolution
from utils import show_img
from argparse import ArgumentParser

def main(save_img=False):
    img = cv2.imread("./photos/profile_photo_501.jpg", 0)
    show_img("Original image", img, save_img=False)


    # High-pass kernel (Laplacian filter)
    edge_threshold = 160   # 邊緣 Threshold 設定 160
    high_pass_kernel = np.array([[1, 1, 1],
                                 [1,-8, 1],
                                 [1, 1, 1]], dtype='int8')
    # edge_threshold = 140  # 邊緣 Threshold 設定 140
    # high_pass_kernel = np.array([[ 0,-1, 0],
    #                              [-1, 4,-1],
    #                              [ 0,-1, 0]], dtype='int8')

    # Low-pass kernel (Mean filter)
    kernel_size = 15
    low_pass_kernel = np.ones((kernel_size, kernel_size), np.float64) / kernel_size ** 2


    # Input High-pass kernel to dft_product
    high_pass_result = dft_product(img, high_pass_kernel, "High_pass", save_img)
    show_img("Result of DFT product, High-pass kernel", high_pass_result, save_img)
    high_pass_result = np.where(high_pass_result > edge_threshold, high_pass_result, 0)
    show_img("Edge of DFT product, High-pass kernel", high_pass_result, save_img)


    # Input Low-pass kernel to dft_product
    low_pass_result = dft_product(img, low_pass_kernel, "Low_pass", save_img)
    show_img("Result of DFT product, Low-pass kernel", low_pass_result, save_img)


    # Input High-pass kernel to convolution
    high_pass_conv_result = convolution(img, high_pass_kernel)
    show_img("Result of Convolution, High-pass kernel", high_pass_conv_result, save_img)
    high_pass_conv_result = np.where(high_pass_conv_result > edge_threshold, high_pass_conv_result, 0)
    show_img("Edge of Convolution, High-pass kernel", high_pass_conv_result, save_img)


    # Input Low-pass kernel to convolution
    low_pass_conv_result = convolution(img, low_pass_kernel)
    show_img("Result of Convolution, Low-pass kernel", low_pass_conv_result, save_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_img", help="Use --save_img if you want to save result images, otherwise don't use it.", action="store_true")
    args = parser.parse_args()
    if args.save_img:
        main(save_img=True)
