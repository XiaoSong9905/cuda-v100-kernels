// Below code are tunned for V-100

#include <cuda_runtime.h>
#include <cstdio>
#include <limits>
#include "../common/common.h"

void transpose_host( const float* h_mtx, float* h_ref_trans_mtx, int nx, int ny )
{
    for ( int iy = 0; iy < ny; ++iy )
    {
        for ( int ix = 0; ix < nx; ++ix )
        {
            h_ref_trans_mtx[ ix * ny + iy ] = h_mtx[ iy * nx + ix ];
        }
    }
}

void init_data( float* h_mtx, int size )
{
    for ( int i = 0; i < size; ++i )
    {
        h_mtx[ i ] = (float)(rand() & 0xFF) / 10.0f;
    }
}

bool check_correctness( const float* h_ref_trans_mtx, const float* h_trans_mtx, int num_elem )
{
    for ( int i = 0; i < size; ++i )
    {
        if ( std::abs( h_ref_trans_mtx[ i ] - h_trans_mtx[ i ] ) > \
             std::numeric_limits<float>::epsilon() )
        {
            return false;
        }
    }
    return true;
}

float compute_bandwidth( int data_in_bytes, float time_in_milliseconds )
{
    return float( data_in_bytes ) / 1e6 / time_in_milliseconds;
}

#define BDIMX 16
#define BDIMY 32
#define PAD 1

__global__ void kernel_copy( float* d_src, float* d_dst, int nx, int ny )
{
    // global idx of 2D matrix
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // direct copy from global memory to global memory
    if ( ix < nx && iy < ny )
    {
        d_dst[ iy * nx + ix ] = d_dst[ iy * nx + ix ];
    }
}

__global__ void kernel_naive_transpose( float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // global idx of 2D matrix
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose by direct copy from global to global
    // coarlesed read
    // stride write
    if ( ix < nx && iy < ny )
    {
        d_dst[ iy * nx + ix ] = d_dst[ iy * nx + ix ];
    }
}

__global__ void kernel_transpose_sharedmem( float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    __shared__ float shared_mem[BDIMY][BDIMX];

    // index notation
    // g for grid level
    // b for block level 
    // i for input
    // o for output

    // index before transpose
    // contigous thread (along threadIdx.x) read contigious global memory location 
    //   and write to contigious shared memory location
    unsigned int i_g_col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i_g_row = blockIdx.y * blockDim.y + threadIdx.y;

    if ( i_g_col < nx && i_g_row < ny )
    {
        // index before transfer
        // contigious thread (i_g_col is the innermost loop) read contigious global memory location
        unsigned int i_g = i_g_row * nx + i_g_col;

        // write to row of shared memory (contigous, threadIdx.x is the innermost loop)
        shared_mem[ threadIdx.y ][ threadIdx.x ] = d_mtx[ i_g ];

        // index after transpose
        // 2D thread index to 1D thread index (how warp is arranged)
        // contigous thread = contigious i_b
        unsigned int i_b = threadIdx.y * blockDim.x + threadIdx.x; 

        // contigious i_b = contigious o_b_col (b.c. use %)
        // contigious thread write to row of global memory (contigous, o_b_col is the innermost loop)
        unsigned int o_b_col = i_b % blockDim.y;
        unsigned int o_b_row = i_b / blockDim.y;
        unsigned int o_g_col = blockDim.y * blockIdx.y + o_b_col;
        unsigned int o_g_row = blockDim.x * blockIdx.x + o_b_row;
        unsigned int o_g = o_g_row * ny + o_g_col;

        __syncthreads();

        // when index to shared memory, swap o_b_col and o_b_row. shared memory is not transposed
        // contigous thread (along o_b_col) read one col of shared memory (first dim indexing)
        d_trans_mtx[ o_g ] = shared_mem[ o_b_col ][ o_b_row ];
    }
}

__global__ void kernel_transpose_sharedmem_pad( float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    __shared__ float shared_mem[BDIMY][BDIMX+PAD];

    // index notation
    // g for grid level
    // b for block level 
    // i for input
    // o for output

    // index before transpose
    // contigous thread (along threadIdx.x) read contigious global memory location 
    //   and write to contigious shared memory location
    unsigned int i_g_col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i_g_row = blockIdx.y * blockDim.y + threadIdx.y;

    if ( i_g_col < nx && i_g_row < ny )
    {
        // index before transfer
        // contigious thread (i_g_col is the innermost loop) read contigious global memory location
        unsigned int i_g = i_g_row * nx + i_g_col;

        // write to row of shared memory (contigous, threadIdx.x is the innermost loop)
        shared_mem[ threadIdx.y ][ threadIdx.x ] = d_mtx[ i_g ];

        // index after transpose
        // 2D thread index to 1D thread index (how warp is arranged)
        // contigous thread = contigious i_b
        unsigned int i_b = threadIdx.y * blockDim.x + threadIdx.x; 

        // contigious i_b = contigious o_b_col (b.c. use %)
        // contigious thread write to row of global memory (contigous, o_b_col is the innermost loop)
        unsigned int o_b_col = i_b % blockDim.y;
        unsigned int o_b_row = i_b / blockDim.y;
        unsigned int o_g_col = blockDim.y * blockIdx.y + o_b_col;
        unsigned int o_g_row = blockDim.x * blockIdx.x + o_b_row;
        unsigned int o_g = o_g_row * ny + o_g_col;

        __syncthreads();

        // when index to shared memory, swap o_b_col and o_b_row. shared memory is not transposed
        // contigous thread (along o_b_col) read one col of shared memory (first dim indexing)
        d_trans_mtx[ o_g ] = shared_mem[ o_b_col ][ o_b_row ];
    }
}

int main( int argc, char** argv )
{
    // set device
    int dev_id = 0;
    if ( argc > 1 )
    {
        dev_id = atoi( argv[ 1 ] );
    }
    CUDA_CHECK( cudaSetDevice( dev_id ) );
    printf("Running on GPU device %d\n", dev_id );

    // Assume matrix size is multiply of block dim
    int nx = 4096;
    int ny = 4096;
    int n_bytes = nx * ny * sizeof(float);
    int n_elems = nx * ny;

    float* h_mtx = (float*)malloc( n_bytes );
    float* h_ref_trans_mtx = (float*)malloc( n_bytes );
    float* h_trans_mtx = (float*)malloc( n_bytes );

    init_data( h_mtx, n_elems );

    float* d_mtx;
    float* d_trans_mtx;
    CUDA_CHECK( cudaMalloc( (float**)&d_mtx, n_bytes ) );
    CUDA_CHECK( cudaMalloc( (float**)&d_trans_mtx, n_bytes ) );

    CUDA_CHECK( cudaMemcpy( d_mtx, h_mtx, n_bytes, cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0.0f, n_bytes ) );

    cudaEvent_t t_start, t_stop;
    CUDA_CHECK( cudaEventCreate( &t_start ) );
    CUDA_CHECK( cudaEventCreate( &t_stop ) );
    float t_milliseconds;
    float bandwidth;
    
    dim3 block( BDIMX, BDIMY );
    dim3 grid( nx / BDIMX, ny / BDIMY );
    printf("Transpose with matrix of size (col %d, row %d)\n", nx, ny );
    printf("Config blockDim (x %d, y %d), gridDim (x %d, y %d)\n", \
        block.x, block.y, grid.x, grid.y );

    // Timing lower bound (copy matrix)
    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_copy<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    bandwidth = compute_bandwidth( n_bytes * 2, t_milliseconds );
    printf("Lower Bound take %.3f sec, Effective bandwidth %.3f GB/sec\n", \
        t_milliseconds / 1e3, bandwidth );

    // Timing upper bound (naive transpose with global memory)
    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_naive_transpose<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) );

    bandwidth = compute_bandwidth( n_bytes * 2, t_milliseconds );
    printf("Upper Bound take %.3f sec, Effective bandwidth %.3f GB/sec\n", \
        t_milliseconds / 1e3, bandwidth );

    // Shared memory
    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) );

    bandwidth = compute_bandwidth( n_bytes * 2, t_milliseconds );
    printf("Shared Mem take %.3f sec, Effective bandwidth %.3f GB/sec\n", \
        t_milliseconds / 1e3, bandwidth );

    // Shared memory + padding
    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_pad<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) );

    bandwidth = compute_bandwidth( n_bytes * 2, t_milliseconds );
    printf("Shared Mem and Pad take %.3f sec, Effective bandwidth %.3f GB/sec\n", \
        t_milliseconds / 1e3, bandwidth );

    // Free host and device memory
    free( h_mtx );
    free( h_ref_trans_mtx );
    free( h_trans_mtx );
    CUDA_CHECK( cudaFree( d_mtx ) );
    CUDA_CHECK( cudaFree( d_trans_mtx ) );

}