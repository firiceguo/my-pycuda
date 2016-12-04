'''
Description:
    This is a serial algorithm of image convolution defination.

Usage:
    $python serial.py

Correctness test:
    Use the conv_cpu function in utils.py to test the correctness of the GPU output.
'''

import numpy as np
import datetime
from utils import conv_cpu

KERNEL_R = 10
KERNEL_L = KERNEL_R * 2 + 1
IMAGE_W = 15


def conv_serial(IMAGE_W):
    '''
    Get the runtime of serial method.
    input:  int IMAGE_W -- the width of image
    output: float secs  -- the runtime usage (microseconds)
    '''
    kernel_cpu = np.random.randn(KERNEL_L, KERNEL_L).astype(np.float32)
    pic_cpu = np.random.randn(IMAGE_W, IMAGE_W).astype(np.float32)

    start = datetime.datetime.now()
    out = conv_cpu(pic_cpu, kernel_cpu, IMAGE_W, KERNEL_R)
    end = datetime.datetime.now()
    secs = (end - start).seconds * 1000 + float('0.' + str((end - start).microseconds)) * 1000

    return(secs)

if __name__ == '__main__':
    secs = conv_serial(IMAGE_W)
    print("IMAGE_W:%d:Time:%f:ms" % (IMAGE_W, secs))
