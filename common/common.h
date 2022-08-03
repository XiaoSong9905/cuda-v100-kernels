#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

void print_dev_prop( )
{
    int num_devs;
    CUDA_CHECK( cudaGetDeviceCount( &num_devs ) );

    int dev_id;
    CUDA_CHECK( cudaGetDevice( &dev_id ) );

    cudaDeviceProp dev_prop;
    CUDA_CHECK( cudaGetDeviceProperties( &dev_prop, dev_id ) );

    int runtime_version; 
    CUDA_CHECK( cudaRuntimeGetVersion( &runtime_version ) );

    printf("\n---Device & Runtime Info--\n");
    printf("runtime version %d\n", runtime_version );
    printf("device name %s\n", dev_prop.name );
    printf("compute capacity %d.%d\n", dev_prop.major, dev_prop.minor );
    printf("global memory size %lu bytes\n", dev_prop.totalGlobalMem );
    printf("global memory clock rate %d\n", dev_prop.memoryClockRate  );
    printf("global memory bus width %d\n", dev_prop.memoryBusWidth );
    printf("global memory bandwidth %f GB/s\n", \
        2.0 * dev_prop.memoryClockRate * ( dev_prop.memoryBusWidth / 8 ) / 1e6 );
    printf("shared memory size %lu bytes\n", dev_prop.sharedMemPerBlock );
    printf("num sm %d\n", dev_prop.multiProcessorCount );
    printf("\n");
}