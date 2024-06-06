#include "../common/utils.hpp"
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

template <typename T>
__global__ void dot_cuda(const T *x, const T *y, T *z, const size_t N)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x)
    {
        z[i] = x[i] * y[i];
    }
}

template <typename T>
__global__ void reduce_cuda(const T *x, T *dx, const size_t N)
{
    extern __shared__ T cache[]; // each block has a shared cache.
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    T y = 0;
    for (int i = tid + bid * blockDim.x; i < N; i += blockDim.x * gridDim.x)
    {
        y += x[i];
    }
    cache[tid] = y;

    __syncthreads();

    // 32 is the default wrap size
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            cache[tid] += cache[tid + offset];
        }
        // __syncwarp();
        __syncthreads();
    }

    y = cache[tid];
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        y += g32.shfl_down(y, offset);
    }
    if (tid == 0)
    {
        dx[bid] = y;
    }
}

template <typename T>
__host__ T dot(const T *x, const T *y, const size_t N)
{
    thrust::universal_vector<T> zv(N);
    auto z = zv.data().get();
    dot_cuda<<<1024, 1024>>>(x, y, z, N);

    const int block_size = 1024;
    const int grid_size = 10240 < ROUND_UP_DIV(N, 1024) ? 10240 : ROUND_UP_DIV(N, 1024);

    thrust::universal_vector<T> dz(grid_size);
    reduce_cuda<<<grid_size, block_size, block_size * sizeof(T)>>>(z, dz.data().get(), N);


    thrust::universal_vector<T> result(1);
    reduce_cuda<<<1, block_size, block_size * sizeof(T)>>>(dz.data().get(), result.data().get(), dz.size());

    CHECK_ERROR(cudaDeviceSynchronize());

    T ret = result[0];
    return ret;
}

int main(int argc, char const *argv[])
{
    const int N = 1e8;
    thrust::device_vector<float> x(N), y(N);

    parallel_for<<<1024, 1024>>>(N, [x = x.data().get(), y = y.data().get()] __device__(int i)
                                 {
        x[i] = 1.5f;
        y[i] = 2.0f; });


    float res = dot(x.data().get(), y.data().get(), x.size());

    std::cout << __FILE__ << "@" << __LINE__ << "\tresult = " << res << std::endl;

    return 0;
}
