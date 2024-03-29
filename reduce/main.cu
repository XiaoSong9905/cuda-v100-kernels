// Below code are tunned for V-100
#include <cstdio>
#include <limits>
#include <omp.h>
#include <chrono>
#include "../common/common.h"

// TODO
// 1. use dynamic parallel to avoid doing any reduction on cpu
// 2. test reference code bandwidth

void init_data( int* ptr, int num_elem )
{
    for ( int i = 0; i < num_elem; ++i )
    {
        *( ptr + i ) = i % 100;
    }
}

// Note: below reduce on host have not been optimized
inline int reduce_host( const int* h_input, int num_elem )
{
    int sum;
    #pragma omp parallel for reduction(+: sum)
    for ( int i = 0; i < num_elem; i++ )
    {
        sum += h_input[ i ];
    }
    return sum;
}

#define BDIM 128
#define WARP_SIZE 32

template< int size >
__device__ __forceinline__ int warp_reduce( int partial_sum )
{
    if ( size >= 32 ) partial_sum += __shfl_xor_sync( 0xffffffff, partial_sum, 16 );
    if ( size >= 16 ) partial_sum += __shfl_xor_sync( 0xffffffff, partial_sum, 8 );
    if ( size >= 8  ) partial_sum += __shfl_xor_sync( 0xffffffff, partial_sum, 4 );
    if ( size >= 4  ) partial_sum += __shfl_xor_sync( 0xffffffff, partial_sum, 2 );
    if ( size >= 2  ) partial_sum += __shfl_xor_sync( 0xffffffff, partial_sum, 1 );
    return partial_sum;
}

template< int num_algocascade >
__global__ void reduce_sharedmem_algocascade_warpshuffle_purewarp( const int* d_input, int* d_output, int num_elem )
{
    __shared__ int shared_mem[ BDIM / WARP_SIZE ];

    // block index
    // used for shared memory indexing
    unsigned int bid = threadIdx.x;

    // global index
    // Each thread block in charge of 4 data block in x direction
    unsigned int gid = blockIdx.x * ( blockDim.x * num_algocascade ) + threadIdx.x;

    // Sequential part of algorithm cascading

    // Thread stride = blockDim.x
    // When testing code, assume data is multiplier of num_algocascade * blockDim.x 
    // When deploying code, hardcode below logic with if branch and partial add to avoid unroll
    int register_array[ num_algocascade ];
    int partial_sum = 0;
    #pragma unroll
    for ( int i = 0; i < num_algocascade; ++i )
    {
        // Use register array to expose more ILP in memory load
        register_array[ i ] = d_input[ gid + i * BDIM ];
    }
    #pragma unroll
    for ( int i = 0; i < num_algocascade; ++i )
    {
        // When hardcode below logic, can use below code to expose more ILP
        // partial_sum = a1 + a2;
        // tmp = a3 + a4;
        // partial_sum += tmp;
        partial_sum += register_array[ i ];
    }

    // Parallel part of algorithm cascading
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    // int lane_id = threadIdx.x & 31;
    // int warp_id = threadIdx.x >> 5;

    // All thread in threads block use warp to reduce
    // This avoid use syncthread() too many time, help scheduler hide latency better
    // May not have significant performence improvement if SM already able to hide latency caused by syncthread
    partial_sum = warp_reduce<WARP_SIZE>( partial_sum );
    
    if ( lane_id == 0 )
        shared_mem[ warp_id ] = partial_sum;
    __syncthreads();
    
    if ( bid < 32 )
    {
        // load shared memory to partial sum
        // we use 0xFFF as mask which means use all participate threads. Thus need to set zero
        if ( bid < BDIM/WARP_SIZE )
            partial_sum = shared_mem[ lane_id ];
        else
            partial_sum = 0;

        partial_sum = warp_reduce<BDIM/WARP_SIZE>( partial_sum );
        if ( bid == 0 )
            d_output[ blockIdx.x ] = partial_sum;
    }
}

template< int num_algocascade >
__global__ void reduce_sharedmem_algocascade_warpshuffle( const int* d_input, int* d_output, int num_elem )
{
    // shared memory size = thread block size != data block size (algo cascading)
    __shared__ int shared_mem[ BDIM ];

    // block index
    // used for shared memory indexing
    unsigned int bid = threadIdx.x;

    // global index
    // Each thread block in charge of 4 data block in x direction
    unsigned int gid = blockIdx.x * ( blockDim.x * num_algocascade ) + threadIdx.x;

    // Sequential part of algorithm cascading

    // Thread stride = blockDim.x
    // When testing code, assume data is multiplier of num_algocascade * blockDim.x 
    // When deploying code, hardcode below logic with if branch and partial add to avoid unroll
    int register_array[ num_algocascade ];
    int partial_sum = 0;
    #pragma unroll
    for ( int i = 0; i < num_algocascade; ++i )
    {
        // Use register array to expose more ILP in memory load
        register_array[ i ] = d_input[ gid + i * BDIM ];
    }
    #pragma unroll
    for ( int i = 0; i < num_algocascade; ++i )
    {
        // When hardcode below logic, can use below code to expose more ILP
        // partial_sum = a1 + a2;
        // tmp = a3 + a4;
        // partial_sum += tmp;
        partial_sum += register_array[ i ];
    }

    shared_mem[ bid ] = partial_sum;
    __syncthreads();

    // Parallell part of algorithm cascading

    // Complete unroll
    // decrease stride with each iteration, reduce divergence & bank conflict
    if ( BDIM >= 1024 ) // will be remove by compiler 
    {
        if ( bid < 512 )
            shared_mem[ bid ] += shared_mem[ bid + 512 ];
        __syncthreads();
    }    
    if ( BDIM >= 512 )
    {
        if ( bid < 256 )
            shared_mem[ bid ] += shared_mem[ bid + 256 ];
        __syncthreads();
    }
    if ( BDIM >= 256 )
    {
        if ( bid < 128 )
            shared_mem[ bid ] += shared_mem[ bid + 128 ];
        __syncthreads();
    }
    if ( BDIM >= 128 )
    {
        if ( bid < 64 )
            shared_mem[ bid ] += shared_mem[ bid + 64 ];
        __syncthreads();
    }
    if ( BDIM >= 64 )
    {
        if ( bid < 32 )
            shared_mem[ bid ] += shared_mem[ bid + 32 ];
        __syncthreads();
    }

    // warp shuffle
    if ( bid < 32 )
    {
        // put shared memory data to register, register reuse, and use warp register shuffle
        partial_sum = shared_mem[ bid ];
        partial_sum = warp_reduce<WARP_SIZE>( partial_sum );

        if ( bid == 0 )
            d_output[ blockIdx.x ] = partial_sum;
    }
}

int main( int argc, char** argv )
{
    int dev_id = 0;
    if ( argc > 1 )
    {
        dev_id = atoi( argv[ 1 ] );
    }
    CUDA_CHECK( cudaSetDevice( dev_id ) );
    print_dev_prop();

    int num_elem = 1024;
    if ( argc > 2 )
    {
        num_elem = atoi( argv[ 2 ] );
    }
    int num_bytes = num_elem * sizeof( int );
    printf("Reduce with array w/ %d num elem\n", num_elem );

    omp_set_num_threads( 20 );
    printf("OpenMP use %d num threads\n", omp_get_num_threads() );

    dim3 block( BDIM );
    dim3 grid( ( num_elem + BDIM - 1 ) / BDIM );
    dim3 grid_unroll4( grid.x / 4 );
    printf("blockDim (x %d), gridDim (x %d), grid_unroll4Dim (x %d)\n", block.x, grid.x, grid_unroll4.x );

    // Allocate resource
    int* h_input = (int*)malloc( num_bytes );
    int* h_output = (int*)malloc( grid.x * sizeof( int ) );
    init_data( h_input, num_elem );

    int* d_input = nullptr;
    int* d_output = nullptr;
    CUDA_CHECK( cudaMalloc( (void**)&d_input, num_bytes ) );
    CUDA_CHECK( cudaMalloc( (void**)&d_output, grid.x * sizeof(int) ));
    CUDA_CHECK( cudaMemcpy( d_input, h_input, num_bytes, cudaMemcpyHostToDevice ));
    CUDA_CHECK( cudaMemset( d_output, 0, grid.x * sizeof(int) ) );

    cudaEvent_t t_gpu_start, t_gpu_end;
    CUDA_CHECK( cudaEventCreate( &t_gpu_start ) );
    CUDA_CHECK( cudaEventCreate( &t_gpu_end ) );
    float t_gpu_milliseconds;
    double t_cpu_seconds;
    float bandwidth_gpu, bandwidth_cpu;

    // Host result
    int cpu_result = reduce_host( h_input, num_elem );
    int gpu_result;
    printf("CPU reduce result %d\n\n", cpu_result );

    // Reduce with shared memory
    {
        CUDA_CHECK( cudaEventRecord( t_gpu_start ) );
        auto t_cpu_start = std::chrono::steady_clock::now();
        reduce_sharedmem_algocascade_warpshuffle<4><<<grid_unroll4, block>>>( d_input, d_output, num_elem );
        CUDA_CHECK( cudaEventRecord( t_gpu_end ) );
        CUDA_CHECK( cudaEventSynchronize( t_gpu_end ) );
        CUDA_CHECK( cudaMemcpy( h_output, d_output, grid_unroll4.x * sizeof( int ), cudaMemcpyDeviceToHost ) );
        gpu_result = reduce_host( h_output, grid_unroll4.x );
        auto t_cpu_end = std::chrono::steady_clock::now();
        
        CUDA_CHECK( cudaEventElapsedTime( &t_gpu_milliseconds, t_gpu_start, t_gpu_end ) );
        std::chrono::duration<double> t_cpu_diff = t_cpu_end - t_cpu_start;
        t_cpu_seconds = t_cpu_diff.count();
    
        bandwidth_gpu = compute_bandwidth_ms( num_bytes, t_gpu_milliseconds );
        bandwidth_cpu = compute_bandwidth_s( num_bytes, t_cpu_seconds );
    
        if ( cpu_result != gpu_result )
            printf("Result mismatch, cpu have %d gpu have %d\n", cpu_result, gpu_result );
        printf("Shared Mem + Unroll + Algo Cascade 4 + Warp Shuffle, GPU Effective bandwidth %.5f GB/sec, CPU Effective bandwidth %.5f GB/sec\n\n", \
            bandwidth_gpu, bandwidth_cpu );    
    }

    // Reduce with nested warp
    {
        CUDA_CHECK( cudaEventRecord( t_gpu_start ) );
        auto t_cpu_start = std::chrono::steady_clock::now();
        reduce_sharedmem_algocascade_warpshuffle_purewarp<4><<<grid_unroll4, block>>>( d_input, d_output, num_elem );
        CUDA_CHECK( cudaEventRecord( t_gpu_end ) );
        CUDA_CHECK( cudaEventSynchronize( t_gpu_end ) );
        CUDA_CHECK( cudaMemcpy( h_output, d_output, grid_unroll4.x * sizeof( int ), cudaMemcpyDeviceToHost ) );
        gpu_result = reduce_host( h_output, grid_unroll4.x );
        auto t_cpu_end = std::chrono::steady_clock::now();
        
        CUDA_CHECK( cudaEventElapsedTime( &t_gpu_milliseconds, t_gpu_start, t_gpu_end ) );
        std::chrono::duration<double> t_cpu_diff = t_cpu_end - t_cpu_start;
        t_cpu_seconds = t_cpu_diff.count();
    
        bandwidth_gpu = compute_bandwidth_ms( num_bytes, t_gpu_milliseconds );
        bandwidth_cpu = compute_bandwidth_s( num_bytes, t_cpu_seconds );

        if ( cpu_result != gpu_result )
            printf("Result mismatch, cpu have %d gpu have %d\n", cpu_result, gpu_result );
        printf("Shared Mem + Algo Cascade 4 + Nested Warp + Warp Shuffle, GPU Effective bandwidth %.5f GB/sec, CPU Effective bandwidth %.5f GB/sec\n\n", \
            bandwidth_gpu, bandwidth_cpu );    
    }

    // Free resource
    free( h_input );
    free( h_output );
    CUDA_CHECK( cudaFree( d_input ) );
    CUDA_CHECK( cudaFree( d_output ) );
    CUDA_CHECK( cudaEventDestroy( t_gpu_start ) );
    CUDA_CHECK( cudaEventDestroy( t_gpu_end ) );
}