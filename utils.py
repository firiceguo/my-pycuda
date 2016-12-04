'''
Description:
    This is a utils package to do the correctness test using CPU, based on convolution definition.

Usage:
    from utils import *

Function introduction:
    conv_cpu:   test the correctness of serial, naive, and redundant boundary algorithm;
    FilterTest: test the correctness of separable filter algorithm.
'''

import numpy as np


def conv_cpu(pic, kernel, IMAGE_W, KERNEL_R):
    '''
    Get the convolution result using CPU.
    input:
        np.ndarray pic[IMAGE_W][IMAGE_W]      -- the input image matrix
        np.ndarray kernel[KERNEL_L][KERNEL_L] -- the input kernel matrix
        int IMAGE_W  -- the width of image
        int KERNEL_R -- the redius of kernel
    output:
        np.ndarray out[IMAGE_W][IMAGE_W] -- the output image matrix
    '''
    out = np.zeros_like(pic)
    for i in xrange(IMAGE_W):
        for j in xrange(IMAGE_W):
            summ = 0
            for x in xrange(-KERNEL_R, KERNEL_R + 1):
                for y in xrange(-KERNEL_R, KERNEL_R + 1):
                    if i + x < 0 or j + y < 0 or i + x > IMAGE_W - 1 or j + y > IMAGE_W - 1:
                        val = 0
                    else:
                        val = pic[i + x][j + y] * kernel[KERNEL_R + x][KERNEL_R + y]
                    summ += val
            out[i][j] = summ
    return(out)


def FilterTest(pic, filterx, filtery, IMAGE_W, KERNEL_R):
    '''
    Get the convolution result using CPU, based on filter algorithm.
    input:
        np.ndarray pic[IMAGE_W][IMAGE_W] -- the input image matrix
        np.ndarray filterx[KERNEL_L]     -- the input row vector
        np.ndarray filtery[KERNEL_L]     -- the input column vector
        int IMAGE_W  -- the width of image
        int KERNEL_R -- the redius of kernel
    output:
        np.ndarray out[IMAGE_W][IMAGE_W] -- the output image matrix
    '''

    out = np.zeros_like(pic)
    tmp = np.zeros_like(pic)

    # row convolution
    for i in xrange(IMAGE_W):
        for j in xrange(IMAGE_W):
            summ = 0
            for k in xrange(-KERNEL_R, KERNEL_R + 1):
                if j + k < 0 or j + k > IMAGE_W - 1:
                    val = 0
                else:
                    val = pic[i][j + k] * filterx[KERNEL_R + k]
                summ += val
            tmp[i][j] += summ

    # column convolution
    for i in xrange(IMAGE_W):
        for j in xrange(IMAGE_W):
            summ = 0
            for k in xrange(-KERNEL_R, KERNEL_R + 1):
                if i + k < 0 or i + k > IMAGE_W - 1:
                    val = 0
                else:
                    val = tmp[i + k][j] * filtery[KERNEL_R + k]
                summ += val
            out[i][j] += summ
    return(out)
