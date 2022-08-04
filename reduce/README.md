# Reduce

## Build & Run
* compile
```shell
make all

make clean
```

* run
```shell
./reduce.out (device_id) (shift)

# example reduce 2^13 on device 1
./reduce.out 1 13
```

## Performence

## Reference
1. UIUC 408 Lecture 17
2. Optimizing parallel reduction in cuda by Mark Harris [link](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
3. Faster Parallel Reductions on Kepler NVIDIA Blog [link](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
4. Professional CUDA C Programming Guide chapter 3, 5
5. Stackoverflow CUDA algorithm cascading [link](https://stackoverflow.com/questions/23232782/cuda-algorithm-cascading)
6. How to optimize in GPU [github](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)