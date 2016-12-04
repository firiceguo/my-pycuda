'''
Description:
    This is the separable filter algorithm of image convolution using CUDA.

Usage:
    $python filter.py

Note:
    When changing the scale of tile, and the width of image,
    please ensure that IMAGE_W >= TILE_W and IMAGE_W % TILE_W == 0,
    or there will be an AssertionError.
    Also, I haven't tried the situation of TILE_W != TILE_H.

Correctness test:
    Use the FilTest function in utils.py to test the correctness of the GPU output.
'''

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import string

KERNEL_R = 10
KERNEL_L = 2 * KERNEL_R + 1
IMAGE_W = 16
TILE_W = TILE_H = 16


def conv_filter(pic, filterx, filtery, IMAGE_W):
    '''
    Get the runtime of separable filter convolution algorithm.
    input:
        np.ndarray pic[IMAGE_W][IMAGE_W] -- the input image matrix
        np.ndarray filterx[KERNEL_L]     -- the input row vector
        np.ndarray filtery[KERNEL_L]     -- the input column vector
        int        IMAGE_W               -- the width of image
    output:
        float secgpu -- the runtime usage (microseconds) 
    '''

    # test the input parameters
    assert pic.dtype == 'float32', 'source image must be float32'
    assert filterx.shape == filtery.shape == (KERNEL_L, ), 'Try changing KERNEL_L'
    assert IMAGE_W >= TILE_W and IMAGE_W % TILE_W == 0, 'Ensure that IMAGE_W >= TILE_W and IMAGE_W % TILE_W == 0'
    assert TILE_W == TILE_H, 'Ensure that TILE_W == TILE_H'

    # convert the scale of inputs, from 2D to 1D
    pic_vector = np.reshape(pic, (-1))
    filterx = np.reshape(filterx, (-1))
    filtery = np.reshape(filtery, (-1))

    # init the intermediate image and output image
    intermediateImage = np.zeros_like(pic_vector)
    destImage = np.zeros_like(pic_vector)

    # init the GPU memory and load the data from CPU
    sourceImage_gpu = drv.mem_alloc(pic_vector.nbytes)
    intermediateImage_gpu = drv.mem_alloc(intermediateImage.nbytes)
    destImage_gpu = drv.mem_alloc(destImage.nbytes)
    filterx_gpu = drv.mem_alloc(filterx.nbytes)
    filtery_gpu = drv.mem_alloc(filtery.nbytes)

    drv.memcpy_htod(sourceImage_gpu, pic_vector)
    drv.memcpy_htod(intermediateImage_gpu, intermediateImage)
    drv.memcpy_htod(destImage_gpu, destImage)
    drv.memcpy_htod(filterx_gpu, filterx)
    drv.memcpy_htod(filtery_gpu, filtery)

    # calculate the grid and block scale according to the width of image and scale of tile
    grids = (IMAGE_W / TILE_W * IMAGE_W / TILE_H, 1)
    blocks = (TILE_W * (TILE_H + 2 * KERNEL_R), 1, 1)

    # run the kernel function and compute the runtime
    start = drv.Event()
    end = drv.Event()
    start.record()
    convolutionRowGPU(intermediateImage_gpu, sourceImage_gpu, filterx_gpu,
                      block=blocks, grid=grids)
    convolutionColGPU(destImage_gpu, intermediateImage_gpu, filtery_gpu,
                      block=blocks, grid=grids)
    end.record()
    end.synchronize()
    secgpu = start.time_till(end)

    # load the output from GPU memory
    out = np.zeros_like(pic_vector)
    drv.memcpy_dtoh(out, destImage_gpu)
    out = np.reshape(out, (IMAGE_W, IMAGE_W))

    sourceImage_gpu.free()
    intermediateImage_gpu.free()
    destImage_gpu.free()
    filterx_gpu.free()
    filtery_gpu.free()

    return(secgpu)


if __name__ == '__main__':
    template = '''
# define IDIVUP(a, b) ( ((a)+1)/(b) + int(((a)+1) %(MOD)s (b) > 0) -1 ) // a is index, b is width, get line index
# define IMUL(a,b) __mul24((a),(b))

// Row convolution filter
__global__ void convolutionRowGPU(
    float out[%(IMAGE_W)s * %(IMAGE_W)s],
    float pic[%(IMAGE_W)s * %(IMAGE_W)s],
    float kernel[%(KERNEL_L)s]
)
{
    /*  Load data as:
        x123x
        x123x
    */
    __shared__ float data[ %(TILE_H)s * (%(TILE_W)s + %(KERNEL_R)s * 2) ];

    // global thread index
    int idx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    // tile index - block base
    int tx = IDIVUP(blockIdx.x, (%(IMAGE_W)s / %(TILE_W)s) );
    int ty = blockIdx.x - IMUL(tx, (%(IMAGE_W)s / %(TILE_W)s));

    // global picture index
    int px = tx * %(TILE_H)s + IDIVUP(threadIdx.x, (%(TILE_W)s + 2 * %(KERNEL_R)s));
    int py = ty * %(TILE_W)s + threadIdx.x - IDIVUP(threadIdx.x, %(TILE_W)s + 2 * %(KERNEL_R)s) * (%(TILE_W)s + 2 * %(KERNEL_R)s) - (%(KERNEL_R)s);

    // data index
    int dx = IDIVUP(threadIdx.x, (%(TILE_W)s + %(KERNEL_R)s * 2));
    int dy = threadIdx.x - IMUL(dx, (%(TILE_W)s + %(KERNEL_R)s * 2));

    // load the pic block into shared memory
    if ((py < 0) || (py > %(IMAGE_W)s - 1)) {
        data[threadIdx.x] = 0;
    }
    else {
        data[threadIdx.x] = pic[px * %(IMAGE_W)s + py];
    }
    __syncthreads();

    // convolution
    float sum = 0;
    if (dy < %(KERNEL_R)s || dy > %(TILE_W)s + %(KERNEL_R)s - 1) {;}
    else {
# pragma unroll
        for (int i = -%(KERNEL_R)s; i <= %(KERNEL_R)s; i++){
            sum += data[dx * (%(TILE_W)s + %(KERNEL_R)s * 2) + dy + i] * kernel[%(KERNEL_R)s + i];
        }
        out[px * %(IMAGE_W)s + py] = sum;
    }
    __syncthreads();
}


// Column convolution filter
__global__ void convolutionColGPU(
    float out[%(IMAGE_W)s * %(IMAGE_W)s],
    float pic[%(IMAGE_W)s * %(IMAGE_W)s],
    float kernel[%(KERNEL_L)s]
)
{
    /*  Load data as:
        xxx
        123
        123
        xxx
    */
    __shared__ float data[ %(TILE_W)s * (%(TILE_H)s + %(KERNEL_R)s * 2) ];

    // global thread index
    int idx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    // tile index - block base
    int tx = IDIVUP(blockIdx.x, (%(IMAGE_W)s / %(TILE_W)s) );
    int ty = blockIdx.x - IMUL(tx, (%(IMAGE_W)s / %(TILE_W)s));

    // global picture index
    int px = tx * %(TILE_H)s + IDIVUP(threadIdx.x, %(TILE_W)s) - %(KERNEL_R)s;
    int py = ty * %(TILE_W)s + threadIdx.x - IDIVUP(threadIdx.x, %(TILE_W)s) * (%(TILE_W)s);

    // data index
    int dx = IDIVUP(threadIdx.x, %(TILE_W)s);
    int dy = threadIdx.x - IMUL(dx, %(TILE_W)s);

    // load the pic block into shared memory
    if ((px < 0) || (px > %(IMAGE_W)s - 1)) {
        data[threadIdx.x] = 0;
    }
    else {
        data[threadIdx.x] = pic[px * %(IMAGE_W)s + py];
    }

    __syncthreads();

    // convolution
    float sum = 0;
    if (dx < %(KERNEL_R)s || dx > %(TILE_H)s + %(KERNEL_R)s - 1) {;}
    else {
# pragma unroll
        for (int i = -%(KERNEL_R)s; i <= %(KERNEL_R)s; i++){
            sum += data[(dx + i) * %(TILE_W)s + dy] * kernel[%(KERNEL_R)s + i];
        }
        out[px * %(IMAGE_W)s + py] = sum;
    }
    __syncthreads();
}
''' % {
        'KERNEL_R': KERNEL_R, 'KERNEL_L': KERNEL_L, 'MOD': '%',
        'IMAGE_W': IMAGE_W, 'TILE_W': TILE_W, 'TILE_H': TILE_H
    }
    module = SourceModule(template)
    convolutionRowGPU = module.get_function('convolutionRowGPU')
    convolutionColGPU = module.get_function('convolutionColGPU')

    pic = np.random.randn(IMAGE_W, IMAGE_W).astype(np.float32)
    filterx = np.random.randn(KERNEL_L).astype(np.float32)
    secs = conv_filter(pic, filterx, filterx, IMAGE_W)
    print("IMAGE_W:%d:Time:%f:ms" % (IMAGE_W, secs))
