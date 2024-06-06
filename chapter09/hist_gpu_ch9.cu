#include "../common/utils.hpp"

__global__ void hstio_(const unsigned char *v,
                       unsigned int *hstio,
                       const size_t N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ unsigned int cache[256];
    cache[tid] = 0;
    __syncthreads();

    for (int i = tid + bid * blockDim.x; i < N; i += blockDim.x * gridDim.x)
    {
        const int y = v[i];
        atomicAdd(&cache[y], 1);
    }

    __syncthreads();

    atomicAdd(&(hstio[tid]), cache[tid]);
}

int main(int argc, char const *argv[])
{
    const int N = 1024 * 1024 * 100; // 100 MB
    thrust::universal_vector<unsigned char> random_v(N);
    thrust::universal_vector<unsigned int> hstio(256, 0);

    // {
    //     unsigned char *v = new unsigned char[N];
    //     cudaMemcpy(random_v.data().get(), v, N, cudaMemcpyHostToDevice);
    //     delete[] v;
    // }

    // block size must be 256

    CUDA_TICK(histo);
    hstio_<<<1024, 256>>>(random_v.data().get(),
                          hstio.data().get(), N);
    CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_TOCK(histo);

    int n = 0;
    for (int i = 0; i < 256; i++)
    {
        n += hstio[i];
        printf("histo[%d] = %d\n", i, hstio[i]);
    }
    printf("all = %d\n", n);
    return 0;
}
