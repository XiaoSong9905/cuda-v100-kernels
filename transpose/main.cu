// Below code are tunned for V-100
#include <cstdio>
#include <limits>
#include "../common/common.h"

// TODO
// 1. use nvprof to analysis num transaction per memory request

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

void init_data( float* ptr, int size )
{
    for ( int i = 0; i < size; ++i )
    {
        ptr[ i ] = (float)(rand() & 0xFF) / 10.0f;
    }
}

bool check_correctness( const float* h_ref_trans_mtx, const float* h_trans_mtx, int num_elem )
{
    for ( int i = 0; i < num_elem; ++i )
    {
        if ( std::abs( h_ref_trans_mtx[ i ] - h_trans_mtx[ i ] ) > \
             std::numeric_limits<float>::epsilon() )
        {
            return false;
        }
    }
    return true;
}

// Note: below code assume blockDim.x & blockDim.y < 32
#define BDIMX 32
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
        d_dst[ iy * nx + ix ] = d_src[ iy * nx + ix ];
    }
}

__global__ void kernel_copy_unroll2( float* d_src, float* d_dst, int nx, int ny )
{
    // Assume nx is even multiplier of blockDim.x
    // Here, we do not change block size, we only change grid size

    // global idx of 2D matrix
    // thread take stride of blockDim.x 
    unsigned int ix = ( 2 * blockIdx.x ) * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // direct copy from global memory to global memory
    if ( ( ix + blockDim.x ) < nx && iy < ny )
    {
        d_dst[ iy * nx + ix              ] = d_src[ iy * nx + ix ];
        d_dst[ iy * nx + ix + blockDim.x ] = d_src[ iy * nx + ix + blockDim.x ];
    }
}

__global__ void kernel_copy_unroll4( float* d_src, float* d_dst, int nx, int ny )
{
    // Assume nx is even multiplier of blockDim.x
    // Here, we do not change block size, we only change grid size

    // global idx of 2D matrix
    // thread take stride of blockDim.x 
    unsigned int ix = ( 4 * blockIdx.x ) * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // direct copy from global memory to global memory
    if ( ( ix + 3 * blockDim.x ) < nx && iy < ny )
    {
        unsigned int srcidx = iy * nx + ix;
        unsigned int dstidx = iy * nx + ix;

        d_dst[ dstidx                  ] = d_src[ srcidx ];
        d_dst[ dstidx + blockDim.x     ] = d_src[ srcidx + blockDim.x ];
        d_dst[ dstidx + 2 * blockDim.x ] = d_src[ srcidx + 2 * blockDim.x ];
        d_dst[ dstidx + 3 * blockDim.x ] = d_src[ srcidx + 3 * blockDim.x ];
    }
}

__global__ void kernel_copy_texturecache( const float* __restrict__ d_src, float* d_dst, int nx, int ny )
{
    // global idx of 2D matrix
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // direct copy from global memory to global memory
    if ( ix < nx && iy < ny )
    {
        d_dst[ iy * nx + ix ] = __ldg( &d_src[ iy * nx + ix ] );
    }
}

__global__ void kernel_naivetranspose( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // global idx of 2D matrix
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose by direct copy from global to global
    // coarlesed read
    // stride write
    if ( ix < nx && iy < ny )
    {
        d_trans_mtx[ ix * ny + iy ] = d_mtx[ iy * nx + ix ];
    }
}

__global__ void kernel_naivetranspose_unroll2( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // Assume nx is even multiplier of blockDim.x
    // Here, we do not change block size, we only change grid size

    // global idx of 2D matrix
    unsigned int ix = ( 2 * blockIdx.x ) * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose by direct copy from global to global
    // coarlesed read
    // stride write
    if ( ( ix + blockDim.x ) < nx && iy < ny )
    {
        d_trans_mtx[ ix * ny + iy ] = d_mtx[ iy * nx + ix ];
        d_trans_mtx[ ix * ny + iy + ny * blockDim.x ] = d_mtx[ iy * nx + ix + blockDim.x ];
    }
}

__global__ void kernel_naivetranspose_unroll4( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // Assume nx is 4 multiplier of blockDim.x
    // Here, we do not change block size, we only change grid size

    // global idx of 2D matrix
    unsigned int ix = ( 4 * blockIdx.x ) * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose by direct copy from global to global
    // coarlesed read
    // stride write
    if ( ( ix + 3 * blockDim.x ) < nx && iy < ny )
    {
        d_trans_mtx[ ix * ny + iy                       ] = d_mtx[ iy * nx + ix ];
        d_trans_mtx[ ix * ny + iy +     ny * blockDim.x ] = d_mtx[ iy * nx + ix + blockDim.x ];
        d_trans_mtx[ ix * ny + iy + 2 * ny * blockDim.x ] = d_mtx[ iy * nx + ix + 2 * blockDim.x ];
        d_trans_mtx[ ix * ny + iy + 3 * ny * blockDim.x ] = d_mtx[ iy * nx + ix + 3 * blockDim.x ];
    }
}

__global__ void kernel_naivetranspose_texturecache( const float* __restrict__ d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // global idx of 2D matrix
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose by direct copy from global to global
    // coarlesed read
    // stride write
    if ( ix < nx && iy < ny )
    {
        d_trans_mtx[ ix * ny + iy ] = __ldg( &d_mtx[ iy * nx + ix ] );
    }
}

__global__ void kernel_transpose_sharedmem( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
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
        // contigious thread = contigious threadIdx.x
        shared_mem[ threadIdx.y ][ threadIdx.x ] = d_mtx[ i_g ];

        // index after transpose
        // 2D thread index to 1D thread index (how warp is arranged)
        // contigous thread = contigious threadIdx.x = contigious i_b
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

__global__ void kernel_transpose_sharedmem_unroll2( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // static 1D shared memory
    __shared__ float shared_mem[BDIMY * (BDIMX * 2)];

    // index before transpose
    // contigous thread (along threadIdx.x) read contigious global memory location 
    //   and write to contigious shared memory location
    unsigned int i_g_col = blockIdx.x * ( 2 * blockDim.x ) + threadIdx.x;
    unsigned int i_g_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_g_col + blockDim.x < nx && i_g_row < ny)
    {
        // index before transfer
        // contigious thread (i_g_col is the innermost loop) read contigious global memory location
        unsigned int i_g = i_g_row * nx + i_g_col;

        // write to row of shared memory (contigous, threadIdx.x is the innermost loop)
        // thread stride size blockDim.x of global and shared memory
        // contigious thread = contigious threadIdx.x = contigious i_g_col = contigious i_g
        unsigned int i_b = threadIdx.y * ( blockDim.x * 2 ) + threadIdx.x;
        shared_mem[ i_b ] = d_mtx[ i_g ];
        shared_mem[ i_b + BDIMX ] = d_mtx[ i_g + BDIMX ];

        // index after transpose
        // 2D thread index to 1D thread index (how warp is arranged)
        // contigous thread = contigious bid1d
        unsigned int bid1d = threadIdx.y * blockDim.x + threadIdx.x;

        // contigious i_b = contigious o_b_col (b.c. use %)
        // contigious thread write to row of global memory (contigous, o_b_col is the innermost loop)
        unsigned int o_b_row = bid1d / blockDim.y;
        unsigned int o_b_col = bid1d % blockDim.y;

        unsigned int o_g_col = blockDim.y * blockIdx.y + o_b_col;
        unsigned int o_g_row = ( 2 * blockDim.x ) * blockIdx.x + o_b_row;
        unsigned int o_g = o_g_row * ny + o_g_col;

        unsigned int o_b = o_b_col * blockDim.x * 2 + o_b_row;

        __syncthreads();

        d_trans_mtx[ o_g ] = shared_mem[ o_b ];
        d_trans_mtx[ o_g + ny * BDIMX ] = shared_mem[ o_b + BDIMX ];
    }
}

// Kernel incorrect
// template< int num_reuse >
// __global__ void kernel_transpose_sharedmem_unrollreuse( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
// {
//     __shared__ float shared_mem[BDIMY][BDIMX];

//     unsigned int i_g_col = blockIdx.x * ( num_reuse * blockDim.x ) + threadIdx.x;
//     unsigned int i_g_row = blockIdx.y * blockDim.y + threadIdx.y;

//     unsigned int i_g = i_g_row * nx + i_g_col;
//     unsigned int bid1d = threadIdx.y * blockDim.x + threadIdx.x;

//     unsigned int o_b_row = bid1d / blockDim.y;
//     unsigned int o_b_col = bid1d % blockDim.y;

//     unsigned int o_g_col = blockDim.y * blockIdx.y + o_b_col;
//     unsigned int o_g_row = ( num_reuse * blockDim.x ) * blockIdx.x + o_b_row;
//     unsigned int o_g = o_g_row * ny + o_g_col;

//     for ( int data_block_i = 0; \
//           data_block_i < num_reuse && i_g_col + blockDim.x * data_block_i < nx && i_g_row < ny; \
//           data_block_i++ )
//     {
//         shared_mem[ threadIdx.y ][ threadIdx.x ] = d_mtx[ i_g + BDIMX * data_block_i ];

//         __syncthreads();

//         d_trans_mtx[ o_g + ny * data_block_i * BDIMX ] = shared_mem[ o_b_col ][ o_b_row ];
//     }
// }

__global__ void kernel_transpose_sharedmem_dynamic( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    // 1D dynamic shared memory
    extern __shared__ float shared_mem[];

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

        // 2D thread index to 1D thread index (how warp is arranged)
        // contigous thread = contigious i_b
        unsigned int i_b = threadIdx.y * blockDim.x + threadIdx.x; 

        // write to row of shared memory (contigous, threadIdx.x is the innermost loop)
        shared_mem[ i_b ] = d_mtx[ i_g ];

        // index after transpose
        // contigious i_b = contigious o_b_col (b.c. use %)
        // contigious thread write to row of global memory (contigous, o_b_col is the innermost loop)
        unsigned int o_b_col = i_b % blockDim.y;
        unsigned int o_b_row = i_b / blockDim.y;
        // index into shared memory. Calculated by swap (o_b_col and o_b_row)
        unsigned int o_b = o_b_col * blockDim.x + o_b_row;

        // contigious thread = contigous o_g_col = contigous o_g
        unsigned int o_g_col = blockDim.y * blockIdx.y + o_b_col;
        unsigned int o_g_row = blockDim.x * blockIdx.x + o_b_row;
        unsigned int o_g = o_g_row * ny + o_g_col;

        __syncthreads();

        d_trans_mtx[ o_g ] = shared_mem[ o_b ];
    }
}

__global__ void kernel_transpose_sharedmem_pad( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
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

template< int num_unroll>
__global__ void kernel_transpose_sharedmem_pad_unroll( const float* d_mtx, float* d_trans_mtx, int nx, int ny )
{
    __shared__ float shared_mem[ BDIMY * ( BDIMX * num_unroll + PAD ) ];

    // index before transpose
    // contigous thread (along threadIdx.x) read contigious global memory location 
    //   and write to contigious shared memory location
    unsigned int i_g_col = blockIdx.x * ( num_unroll * blockDim.x ) + threadIdx.x;
    unsigned int i_g_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_g_col + blockDim.x < nx && i_g_row < ny)
    {
        // index before transfer
        // contigious thread (i_g_col is the innermost loop) read contigious global memory location
        unsigned int i_g = i_g_row * nx + i_g_col;

        // write to row of shared memory (contigous, threadIdx.x is the innermost loop)
        // thread stride size blockDim.x of global and shared memory
        // contigious thread = contigious threadIdx.x = contigious i_g_col = contigious i_g
        unsigned int i_b = threadIdx.y * ( blockDim.x * num_unroll + PAD ) + threadIdx.x;
        #pragma unroll
        for ( int unroll_i = 0; unroll_i < num_unroll; ++unroll_i )
        {
            shared_mem[ i_b + BDIMX * unroll_i ] = d_mtx[ i_g + BDIMX * unroll_i ];
        }

        // index after transpose
        // 2D thread index to 1D thread index (how warp is arranged)
        // contigous thread = contigious bid1d
        unsigned int bid1d = threadIdx.y * blockDim.x + threadIdx.x;

        // contigious i_b = contigious o_b_col (b.c. use %)
        // contigious thread write to row of global memory (contigous, o_b_col is the innermost loop)
        unsigned int o_b_row = bid1d / blockDim.y;
        unsigned int o_b_col = bid1d % blockDim.y;

        unsigned int o_g_col = blockDim.y * blockIdx.y + o_b_col;
        unsigned int o_g_row = ( num_unroll * blockDim.x ) * blockIdx.x + o_b_row;
        unsigned int o_g = o_g_row * ny + o_g_col;

        unsigned int o_b = o_b_col * ( blockDim.x * num_unroll + PAD ) + o_b_row;

        __syncthreads();

        #pragma unroll
        for ( int unroll_i = 0; unroll_i < num_unroll; ++unroll_i )
        {
            d_trans_mtx[ o_g + ny * BDIMX * unroll_i ] = shared_mem[ o_b + BDIMX * unroll_i ];
        }
    }
}

__global__ void kernel_transpose_sharedmem_pad_texturecache( const float* __restrict__ d_mtx, float* d_trans_mtx, int nx, int ny )
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
        shared_mem[ threadIdx.y ][ threadIdx.x ] = __ldg( &d_mtx[ i_g ] ); 

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
    print_dev_prop();

    // Assume matrix size is multiply of block dim
    int nx = 1 << 13;
    int ny = 1 << 13;
    int n_bytes = nx * ny * sizeof(float);
    int n_elems = nx * ny;

    float* h_mtx = (float*)malloc( n_bytes );
    float* h_ref_trans_mtx = (float*)malloc( n_bytes );
    float* h_trans_mtx = (float*)malloc( n_bytes );

    init_data( h_mtx, n_elems );
    transpose_host( h_mtx, h_ref_trans_mtx, nx, ny );

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
    dim3 grid_unroll2( nx / BDIMX / 2, ny / BDIMY ); // setting data block size different from thread block size through grid
    dim3 grid_unroll4( nx / BDIMX / 4, ny / BDIMY );
    printf("Transpose with matrix of size (col %d, row %d)\n\n", nx, ny );

#if 0
    // copy with cuda memcpy
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    CUDA_CHECK( cudaMemcpy( d_trans_mtx, d_mtx, n_bytes, cudaMemcpyDeviceToDevice ) );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("cudaMemcpy Effective bandwidth %.5f GB/sec\n\n", bandwidth );

    // copy matrix
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_copy<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Copy blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // copy matrix with texture cache
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_copy_texturecache<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Copy + Texture cache blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // copy matrix with unroll 2 (expose more ILP)
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_copy_unroll2<<<grid_unroll2, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Copy + Unroll 2 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll2.x, grid_unroll2.y, bandwidth );

    // copy matrix with unroll 2 (expose more ILP)
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_copy_unroll4<<<grid_unroll4, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Copy + Unroll 4 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll4.x, grid_unroll4.y, bandwidth );

    // Naive transpose
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_naivetranspose<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Naive transpose blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // Naive transpose with texture cache
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_naivetranspose_texturecache<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Naive transpose + Texture cache blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // Naive transpose with unroll 2
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_naivetranspose_unroll2<<<grid_unroll2, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Naive transpose + Unroll 2 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll2.x, grid_unroll2.y, bandwidth );

    // Naive transpose with unroll 4
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_naivetranspose_unroll4<<<grid_unroll4, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Naive transpose + Unroll 4 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll4.x, grid_unroll4.y, bandwidth );

    // Shared memory
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // Shared memory + dynamic
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_dynamic<<<grid, block, BDIMX * BDIMY * sizeof(float)>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem + Dynamic blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // Shared memory + unroll 2
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_unroll2<<<grid_unroll2, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem + Unroll 2 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll2.x, grid_unroll2.y, bandwidth );

    // Kernel incorrect, currently disabled
    // shared memory + unroll 2 + reuse
    // CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    // memset( h_trans_mtx, 0, n_bytes );

    // CUDA_CHECK( cudaEventRecord( t_start ) );
    // kernel_transpose_sharedmem_unrollreuse<2><<<grid_unroll2, block>>>( d_mtx, d_trans_mtx, nx, ny );
    // CUDA_CHECK( cudaEventRecord( t_stop ) );
    // CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    // CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    // CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    // if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
    //     printf("Transpose Result Incorrect\n");

    // bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    // printf("Share Mem + Unroll 2 + Reuse blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
    //     block.x, block.y, grid_unroll2.x, grid_unroll2.y, bandwidth );

    // Shared memory + padding
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_pad<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem + Pad blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

    // shared memory + padding + unroll 2
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_pad_unroll<2><<<grid_unroll2, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem + Pad + Unroll 2 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll2.x, grid_unroll2.y, bandwidth );

    // Shared memory + padding + texture cache
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_pad_texturecache<<<grid, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem + Pad + Texture Cache blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid.x, grid.y, bandwidth );

#endif

    // shared memory + padding + unroll 4
    CUDA_CHECK( cudaMemset( d_trans_mtx, 0, n_bytes) );
    memset( h_trans_mtx, 0, n_bytes );

    CUDA_CHECK( cudaEventRecord( t_start ) );
    kernel_transpose_sharedmem_pad_unroll<4><<<grid_unroll4, block>>>( d_mtx, d_trans_mtx, nx, ny );
    CUDA_CHECK( cudaEventRecord( t_stop ) );
    CUDA_CHECK( cudaEventSynchronize( t_stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &t_milliseconds, t_start, t_stop ) );

    CUDA_CHECK( cudaMemcpy( h_trans_mtx, d_trans_mtx, n_bytes, cudaMemcpyDeviceToHost ) );
    if ( !check_correctness( h_trans_mtx, h_ref_trans_mtx, n_elems ) )
        printf("Transpose Result Incorrect\n");

    bandwidth = compute_bandwidth_ms( n_bytes * 2, t_milliseconds );
    printf("Share Mem + Pad + Unroll 4 blockDim (x %d, y %d), gridDim (x %d, y %d) Effective bandwidth %.5f GB/sec\n\n", \
        block.x, block.y, grid_unroll4.x, grid_unroll4.y, bandwidth );

    // Free host and device memory
    free( h_mtx );
    free( h_ref_trans_mtx );
    free( h_trans_mtx );
    CUDA_CHECK( cudaFree( d_mtx ) );
    CUDA_CHECK( cudaFree( d_trans_mtx ) );
    CUDA_CHECK( cudaEventDestroy( t_start ) );
    CUDA_CHECK( cudaEventDestroy( t_end ) );
}