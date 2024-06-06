#include "../common/utils.hpp"
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>

#include "../common/image.h"

const int DIM = 10000;

struct cuComplex
{
    float r;
    float i;
    __device__ __host__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ __host__ float magnitude2(void)
    {
        return r * r + i * i;
    }
    __device__ __host__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ __host__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ __host__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

void kernel_cpu(unsigned char *ptr)
{
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 4 + 0] = 255 * juliaValue;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

void __global__ kernel(unsigned char *ptr)
{
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = offset % DIM;
    const int y = offset / DIM;
    const int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(int argc, char const *argv[])
{
    IMAGE bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();

    TICK(kernel_cpu);
    kernel_cpu(ptr);
    TOCK(kernel_cpu);


    bitmap.show_image();

    thrust::device_vector<unsigned char> dv(bitmap.image_size());

    CUDA_TICK(kernel_gpu);
    kernel<<<ROUND_UP_DIV(DIM*DIM,1024), 1024>>>(dv.data().get());
    cudaDeviceSynchronize();
    CUDA_TOCK(kernel_gpu);

    memset(bitmap.get_ptr(), 0, bitmap.image_size());

    bitmap.show_image();

    CHECK_ERROR(cudaMemcpy(bitmap.get_ptr(),
                           dv.data().get(),
                           dv.size(),
                           cudaMemcpyDeviceToHost));
    
    bitmap.show_image();
    return 0;
}
