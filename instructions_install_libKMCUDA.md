# Instructions for installing libKMCUDA 
libKMCUDA provides the GPU verison of kmeans algorithm used by our proposed defense Feature Compression (FeCo). 
If you want to use CPU, you do not need to install libKMCUDA, since FeCo will use another kmeans package kmeans-pytorch that is easier to install.

## step 1: check cmake version
`cmake --version`. The version should be 3.2 or higher. If not, you should upgrade it.

## step 2: setup CUDA path
`export CUDA_TOOLKIT_ROOT_DIR=XX`. Replace XX with the actual path of your cuda toolkit (usually /usr/local/cuda)

`echo $CUDA_TOOLKIT_ROOT_DIR`

## step 3: check the computation power (CUDA_ARCH) of your GPU. 
See [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

For example, for RTX 3090 has computation power (CUDA_ARCH) of 86.

## step 4: install
```
CUDA_ARCH=XX pip install git+https://github.com/src-d/kmcuda.git#subdirectory=src
```
Replace XX with the CUDA_ARCH obtained in step-3.

If you have problem in accessing github, try to download [the kmcuda repo](https://github.com/src-d/kmcuda) manually, un-compress it, enter the directory, and run:
```
cd src
CUDA_ARCH=XX pip install .
```
note: do not omit the '.' in the last command which means installing in the current directory.

## step 5: testing installation
`python test_kmcuda_install.py`
