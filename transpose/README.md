# Matrix Transpose
## Build & Run
* compile 
```shell
make all

make clean
```

* run 
```shell
./transpose.out (device_id)
```

## Performence
* Settings

Matrix size 8192 * 8192

Matrix dtype float




* fix block config (x32, y32) test different methods

| method                           | grid config    | effective bandwidth (GB/s) | effective bandwidth / max bandwidth (%) | Num instruction / global memory load | num instruction / global memory store | num  bank conflict / shared memory load | num bank conflict / shared memory store |
| -------------------------------- | -------------- | -------------------------- | --------------------------------------- | ------------------------------------ | ------------------------------------- | --------------------------------------- | --------------------------------------- |
| cudaMemcpy                       |                | 742.61755                  |                                         |                                      |                                       |                                         |                                         |
| Copy                             | (x 256, y 256) | 729.19055                  |                                         |                                      |                                       |                                         |                                         |
| Copy + Texture Cache             | (x 256, y 256) | 725.15631                  |                                         |                                      |                                       |                                         |                                         |
| Copy + Unroll 2                  | (x 128, y 256) | 767.62518                  |                                         |                                      |                                       |                                         |                                         |
| Copy + Unroll 4                  | (x 64, y 256)  | 791.97583                  |                                         |                                      |                                       |                                         |                                         |
| Naive Transpose                  | (x 256, y 256) | 186.44666                  |                                         |                                      |                                       |                                         |                                         |
| Naive Transpose + Texture Cache  | (x 256, y 256) | 186.11572                  |                                         |                                      |                                       |                                         |                                         |
| Naive Transpose + Unroll 2       | (x 128, y 256) | 186.64578                  |                                         |                                      |                                       |                                         |                                         |
| Naive Transpose + Unroll 4       | (x 64, y 256)  | 186.31415                  |                                         |                                      |                                       |                                         |                                         |
| Share Mem                        | (x 256, y 256) | 468.11429                  |                                         |                                      |                                       |                                         |                                         |
| Shared Mem + Dynamic             | (x 256, y 256) | 468.11429                  |                                         |                                      |                                       |                                         |                                         |
| Shared Mem + Unroll 2            | (x 128, y 256) | 546.70282                  |                                         |                                      |                                       |                                         |                                         |
| Shared Mem + Pad                 | (x 256, y 256) | 624.89630                  |                                         |                                      |                                       |                                         |                                         |
| Shared Mem + Pad + Unroll 2      | (x 128, y 256) | 656.18024                  |                                         |                                      |                                       |                                         |                                         |
| Shared Mem + Pad + Unroll 4      | (x 64, y 256)  | 657.82684                  |                                         |                                      |                                       |                                         |                                         |
| Shared Mem + Pad + Texture Cache | (x 256, y 256) | 622.66980                  |                                         |                                      |                                       |                                         |                                         |




* fix method (Shared Mem + Pad + Unroll 4) , change block setting

| grid config    | block config | effective bandwidth GB/s | effective bandwidth / max bandwidth (%) |
| -------------- | ------------ | ------------------------ | --------------------------------------- |
| (x 64, y 256)  | (x 32, y 32) | 651.28943                |                                         |
| (x 128, y 512) | (x 16, y 16) | 587.10864                |                                         |
| (x 64, y 512)  | (x 32, y 16) | 591.08002                |                                         |
| (x 128, y 256) | (x 16, y 32) | 644.88068                |                                         |
|                |              |                          |                                         |
|                |              |                          |                                         |
|                |              |                          |                                         |
|                |              |                          |                                         |
|                |              |                          |                                         |



## References
1. Professional CUDA C Programming chapter 5 [link to github code](https://github.com/deeperlearning/professional-cuda-c-programming)