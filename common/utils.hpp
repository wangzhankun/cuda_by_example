#include "global.hpp"


template <typename Func>
__global__ void parallel_for(int N, Func func)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += blockDim.x * gridDim.x)
    {
        func(n);
    }
}

