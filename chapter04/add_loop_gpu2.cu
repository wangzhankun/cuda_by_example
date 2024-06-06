#include "../common/utils.hpp"
#include <thrust/universal_vector.h>

int main(int argc, char const *argv[])
{
    const int N = 10;
    thrust::universal_vector<int> a(N), b(N), c(N);
    parallel_for<<<1, 32>>>(N, [a=a.data(), b=b.data()] __device__ (int i)
    {
        a[i] = -i;
        b[i] = i * i;
    });


    parallel_for<<<1, 32>>>(N, [a=a.data(), b = b.data(), c = c.data()] __device__ (int i){
        c[i] = a[i] + b[i];
    });

    cudaDeviceSynchronize();

    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
    return 0;
}

