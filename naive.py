'''
Description:
    This is a naive algorithm of image convolution using CUDA.

Usage:
    $python naive.py

Correctness test:
    Use the conv_cpu function in utils.py to test the correctness of the GPU output.
'''

import pycuda.driver as drv
from pycuda import compiler, gpuarray, tools, autoinit
from pycuda.compiler import SourceModule
import numpy as np

KERNEL_R = 10
KERNEL_L = KERNEL_R * 2 + 1
IMAGE_W = 15


def Naive(IMAGE_W, knl_template):
    '''
    Get the runtime of conventional convolution algorithm.
    input:
        int    IMAGE_W      -- the width of image
        string knl_template -- a python string of kernel function written by CUDA C
    output:
        float secs  -- the runtime usage (microseconds)
    '''

    # random init image and kernel matrix
    pic_cpu = np.random.randn(IMAGE_W, IMAGE_W).astype(np.float32)
    kernel_cpu = np.random.randn(KERNEL_L, KERNEL_L).astype(np.float32)

    # convert the scale of inputs, from 2D to 1D
    kernel_s = np.reshape(kernel_cpu, (-1))
    pic_s = np.reshape(pic_cpu, (-1))
    out_s = np.zeros_like(pic_s)

    # init the GPU memory and load the data from CPU
    kernel_gpu = drv.mem_alloc(kernel_s.nbytes)
    pic_gpu = drv.mem_alloc(pic_s.nbytes)
    out_gpu = drv.mem_alloc(pic_s.nbytes)

    drv.memcpy_htod(kernel_gpu, kernel_s)
    drv.memcpy_htod(pic_gpu, pic_s)
    drv.memcpy_htod(out_gpu, out_s)

    # build the kernel function
    mod = compiler.SourceModule(knl_template)
    niv_conv = mod.get_function("niv_conv")

    # run the kernel function and compute the runtime
    start = drv.Event()
    end = drv.Event()
    start.record()
    niv_conv(out_gpu, kernel_gpu, pic_gpu,
             block=(512, 1, 1), grid=(1, 1))
    end.record()
    end.synchronize()
    secs = start.time_till(end)

    # load the output from GPU memory
    outgpu = np.zeros_like(out_s)
    drv.memcpy_dtoh(outgpu, out_gpu)
    outgpu = np.reshape(outgpu, (IMAGE_W, IMAGE_W))

    kernel_gpu.free()
    pic_gpu.free()
    out_gpu.free()

    return(secs)


if __name__ == '__main__':
    knl_template = """
__global__ void niv_conv(
    float out[%(IMAGE_W)s * %(IMAGE_W)s],
    float kernel[%(KERNEL_R)s * %(KERNEL_R)s],
    float pic[%(IMAGE_W)s * %(IMAGE_W)s])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float value = 0;

    while(idx < %(IMAGE_W)s * %(IMAGE_W)s){
        float sum = 0;
        int dx = idx / %(IMAGE_W)s;
        int dy = idx %(MOD)s (%(IMAGE_W)s);
#pragma unroll
        for(int i = -%(KERNEL_R)s; i <= %(KERNEL_R)s; i++){
#pragma unroll
            for(int j = -%(KERNEL_R)s; j <= %(KERNEL_R)s; j++){
                if ((dx + i < 0) || (dx + i >= %(IMAGE_W)s)){
                    value = 0;
                }
                else if ((dy + j < 0) || (dy + j >= %(IMAGE_W)s)){
                    value = 0;
                }
                else{
                    value = pic[(dx + i) * %(IMAGE_W)s + (dy + j)] *
                            kernel[%(KERNEL_L)s * (%(KERNEL_R)s + i) + %(KERNEL_R)s + j];
                }
                sum += value;
            }
        }
        out[idx] = sum;
        idx += blockDim.x * gridDim.x;
    }
}
""" % {
        'KERNEL_R': KERNEL_R, 'KERNEL_L': KERNEL_L, 'IMAGE_W': IMAGE_W, 'MOD': '%'
    }
    secs = Naive(IMAGE_W, knl_template)
    print("IMAGE_W:%d:Time:%f:ms" % (IMAGE_W, secs))
