import numpy as np


def conv_cpu(pic, kernel, IMAGE_W, KERNEL_R):
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
    out = np.zeros_like(pic)
    tmp = np.zeros_like(pic)
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
