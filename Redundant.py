import pycuda.driver as drv
from pycuda import compiler, autoinit
from pycuda.compiler import SourceModule
import numpy as np

KERNEL_R = 1
KERNEL_L = KERNEL_R * 2 + 1
IMAGE_W = 16


def getDataLenPerBlk(grid, IMAGE_W):
    blockNum = grid[0]
    n = IMAGE_W / blockNum + int(IMAGE_W % blockNum > 0)
    DataLenPerBlk = IMAGE_W * n
    return(DataLenPerBlk)


def Redundant(IMAGE_W, knl_template):
    kernel_cpu = np.random.randn(KERNEL_L, KERNEL_L).astype(np.float32)
    pic_cpu = np.random.randn(IMAGE_W, IMAGE_W).astype(np.float32)

    kernel_s = np.reshape(kernel_cpu, (-1))
    pic_s = np.reshape(pic_cpu, (-1))
    out_s = np.zeros_like(pic_s)

    kernel_gpu = drv.mem_alloc(kernel_s.nbytes)
    pic_gpu = drv.mem_alloc(pic_s.nbytes)
    out_gpu = drv.mem_alloc(pic_s.nbytes)

    drv.memcpy_htod(kernel_gpu, kernel_s)
    drv.memcpy_htod(pic_gpu, pic_s)
    drv.memcpy_htod(out_gpu, out_s)

    mod = compiler.SourceModule(knl_template)

    redd_conv = mod.get_function("redd_conv")

    start = drv.Event()
    end = drv.Event()
    start.record()
    redd_conv(out_gpu, kernel_gpu, pic_gpu,
              block=block, grid=grid)
    end.record()
    end.synchronize()
    secs = start.time_till(end)

    outgpu = np.zeros_like(out_s)
    drv.memcpy_dtoh(outgpu, out_gpu)
    outgpu = np.reshape(outgpu, (IMAGE_W, IMAGE_W))

    kernel_gpu.free()
    pic_gpu.free()
    out_gpu.free()

    return(secs)


if __name__ == '__main__':
    block = (1024, 1, 1)
    grid = (16, 1)
    # IMAGE_W = 512

    ParaTestPass = (IMAGE_W >= grid[0]) and (IMAGE_W % grid[0] == 0)
    assert ParaTestPass, "Please retry with IMAGE_W >= grid[0] and (IMAGE_W mod grid[0]) == 0."

    DataLenPerBlk = getDataLenPerBlk(grid, IMAGE_W)
    knl_template = """
#define IDIVUP(a, b) ( (a+1)/(b) + int((a+1) %(MOD)s (b) > 0) -1 ) // get line idx in block
#define BLKNUM (%(DataLenPerBlk)s / %(IMAGE_W)s)

__global__ void redd_conv(
    float out[%(IMAGE_W)s * %(IMAGE_W)s],
    float kernel[%(KERNEL_R)s * %(KERNEL_R)s],
    float pic[%(IMAGE_W)s * %(IMAGE_W)s])
{
    int idx = threadIdx.x + blockIdx.x * %(DataLenPerBlk)s;

    __shared__ float data[%(DataLenPerBlk)s + %(IMAGE_W)s * %(KERNEL_R)s * 2];

    int px = blockIdx.x * BLKNUM + IDIVUP(threadIdx.x, %(IMAGE_W)s) - %(KERNEL_R)s;
    int py = threadIdx.x %(MOD)s %(IMAGE_W)s;

    if (threadIdx.x >= %(DataLenPerBlk)s + %(IMAGE_W)s * %(KERNEL_R)s * 2){;}
    else if (px < 0){ // global top
        data[threadIdx.x] = 0;
    }
    else if (px * %(IMAGE_W)s + py > %(IMAGE_W)s * %(IMAGE_W)s - 1){ // global bottom
        data[threadIdx.x] = 0;
    }
    else{
        data[threadIdx.x] = pic[px * %(IMAGE_W)s + py];
    }
    __syncthreads();

    float sum = 0;
    float value = 0;
    int dx = IDIVUP(threadIdx.x, %(IMAGE_W)s) + %(KERNEL_R)s;
    int dy = threadIdx.x %(MOD)s (%(IMAGE_W)s);
#pragma unroll
    for(int i = -%(KERNEL_R)s; i <= %(KERNEL_R)s; i++){
#pragma unroll
        for(int j = -%(KERNEL_R)s; j <= %(KERNEL_R)s; j++){
            if ((dy + j < 0) || (dy + j >= %(IMAGE_W)s)){ // left and right
                value = 0;
            }
            else if ((dx + i) * %(IMAGE_W)s + (dy + j) >= %(DataLenPerBlk)s + %(IMAGE_W)s * %(KERNEL_R)s * 2){
                value = 0;
            }
            else{
                value = data[(dx + i) * %(IMAGE_W)s + (dy + j)] *
                        kernel[%(KERNEL_L)s * (%(KERNEL_R)s + i) + %(KERNEL_R)s + j];
            }
            sum += value;
        }
    }
    out[idx] = sum;
}
""" % {
        'KERNEL_R': KERNEL_R, 'KERNEL_L': KERNEL_L, 'IMAGE_W': IMAGE_W,
        'MOD': '%', 'DataLenPerBlk': DataLenPerBlk
    }
    secs = Redundant(IMAGE_W, knl_template)
    print("IMAGE_W:%d:Time:%f:ms" % (IMAGE_W, secs))
