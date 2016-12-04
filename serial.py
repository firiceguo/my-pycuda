import numpy as np
import datetime
from utils import conv_cpu

KERNEL_R = 10
KERNEL_L = KERNEL_R * 2 + 1
IMAGE_W = 15


def test(IMAGE_W):
    kernel_cpu = np.random.randn(KERNEL_L, KERNEL_L).astype(np.float32)
    pic_cpu = np.random.randn(IMAGE_W, IMAGE_W).astype(np.float32)

    start = datetime.datetime.now()
    out = conv_cpu(pic_cpu, kernel_cpu, IMAGE_W, KERNEL_R)
    end = datetime.datetime.now()
    secs = (end - start).seconds * 1000 + float('0.' + str((end - start).microseconds)) * 1000

    return(secs)

if __name__ == '__main__':
    secs = test(IMAGE_W)
    print("IMAGE_W:%d:Time:%f:ms" % (IMAGE_W, secs))
