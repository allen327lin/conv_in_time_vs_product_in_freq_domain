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
from math import floor, ceil

def resize_kernel(kernel, target_side_length):

    print(kernel.shape[0])

    if kernel.shape[0] != kernel.shape[1]:
        return -1
    else:
        kernel_side_length = kernel.shape[0]

    kernel = np.array(kernel, dtype='int8')

    kernel_resized = np.zeros((target_side_length, target_side_length), dtype=np.float64)
    for x in range(target_side_length):
        for y in range(target_side_length):
            x_ratio = x / (target_side_length - 1) * (kernel_side_length - 1)
            y_ratio = y / (target_side_length - 1) * (kernel_side_length - 1)
            x_after_y_floor = kernel[floor(x_ratio)][floor(y_ratio)] * (1 - (x_ratio - floor(x_ratio))) + \
                              kernel[ceil(x_ratio)][floor(y_ratio)] * (1 - (ceil(x_ratio) - x_ratio))
            x_after_y_ceil = kernel[floor(x_ratio)][ceil(y_ratio)] * (1 - (x_ratio - floor(x_ratio))) + \
                             kernel[ceil(x_ratio)][ceil(y_ratio)] * (1 - (ceil(x_ratio) - x_ratio))
            result = (x_after_y_floor * (1 - (y_ratio - floor(y_ratio))) +
                      x_after_y_ceil * (1 - (ceil(y_ratio) - y_ratio)))
            kernel_resized[x][y] = result / 4
            print(x, y)

    print(kernel_resized)
    print(kernel_resized.shape)

    return kernel_resized


if __name__ == "__main__":
    input_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype='int8')
    side_length = 256
    resize_kernel(input_kernel, side_length)