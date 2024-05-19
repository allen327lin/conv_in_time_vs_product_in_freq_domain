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


def show_img(title, img, save_img=False):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 400, 400)
    cv2.imshow(title, img)
    if save_img:
        cv2.imwrite("./photos/results/" + title + ".png", img)
    return 0


def show_fft(title, fft, save_img=False):
    fft = np.log(np.abs(fft) + 1)
    fft = fft / fft.max() * 255.0
    fft = np.uint8(fft)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 400, 400)
    cv2.imshow(title, fft)
    if save_img:
        cv2.imwrite("./photos/results/" + title + ".png", fft)

    return 0


def normalization(arr):
    min_v = np.min(arr)
    max_v = np.max(arr)

    arr = (arr - min_v) / (max_v - min_v)
    arr = (arr * 255).astype(np.uint8)

    return arr