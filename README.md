## About

I'm trying to use PyCUDA and playing with image convolution algorithems. 


## Test Environment

- [CUDA 8.0](https://developer.nvidia.com/cuda-downloads)

- [PyCUDA 2016.1.2](https://pypi.python.org/pypi/pycuda)

- Python 2.7.12

- NumPy

- Linux Mint 18 (4.4.0-47-generic)

- NVIDIA GTX 950M (DDR5, 4096M)

- Intel Core i7-6700HQ Processor


## Environment setup hints

Basiclly, I followed the [official quick start guide](https://developer.nvidia.com/compute/cuda/8.0/prod/docs/sidebar/CUDA_Quick_Start_Guide-pdf) to install the CUDA Toolkit 8.0

- Use the NVIDIA driver instead the default opensource driver

- Download the `deb(local)` file for `Linux - Ubuntu - 16.04` system from [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads)

- Install the `deb` package, then reboot the system to load the NVIDIA drivers.

- Because Linux Mint is not the official support system, there will be some errors after run the `$ cuda-install-samples-8.0.sh ~`:

1. Fail to run `make` command:  

The error is:

```
>>> WARNING - libGL.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<
>>> WARNING - libGLU.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<
>>> WARNING - libX11.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<
```

To avoid this error, we should add a line in `~/.bashrc` if you are using bash as your default shell:

```
export GLPATH=/usr/lib
```

2. No 'nvcc' error:

The error is:

```
No command 'nvcc' found
```

To avoid this error, we should add a line in `~/.bashrc` if you are using bash as your default shell:

```
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
```

3. More lines should be included in `~/.bashrc`:

```
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_ROOT=/usr/local/cuda-8.0
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib
export CPATH=$CPATH:/usr/local/cuda-8.0/include
export CUDA_INC_DIR=/usr/local/cuda-8.0/bin:$CUDA_INC_DIR
```


## Algorithms

1. Serial Algorithm: `serial.py`

2. Conventional Algorithm: `naive.py`

3. Redundant Boundary Computation Algorithm: `Redundant.py`

4. Separable Filter Algorithm: `filter.py`

5. Some useful utils: `utils.py` 


## Other files

- The `log` directory: There are some runtime information of each algorithem in this directory.

- The `plot` directory: There are some scripts to make a visualization of those logs in the `log` directory and some sample plots.


## References

- [Examples of PyCuda usage](https://wiki.tiker.net/PyCuda/Examples)

- [S. Yu, M. Clement, Q. Snell, and B. Morse. Parallel algorithms for image convolution y. tc (2m2+ 1), 2:2, 1998.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.50.7783)

- [V. Podlozhnyuk. Image convolution with cuda. NVIDIA Corporation white paper, June, 2097(3), 2007.](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/convolutionSeparable/doc/convolutionSeparable.pdf)
