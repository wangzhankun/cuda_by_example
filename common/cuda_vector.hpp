#include "global.hpp"

template <typename T>
__global__ void vector_add(T *x, const T *y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

template <typename T>
__global__ void vector_sub(T *x, const T *y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] - y[n];
    }
}

template <typename T>
__global__ void vector_mul(T *x, const T *y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] * y[n];
    }
}

template <typename T>
__global__ void vector_div(T *x, const T *y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] / y[n];
    }
}

template <typename T>
__global__ void vector_add(T *x, const T y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y;
    }
}

template <typename T>
__global__ void vector_sub(T *x, const T y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] - y;
    }
}

template <typename T>
__global__ void vector_mul(T *x, const T y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] * y;
    }
}

template <typename T>
__global__ void vector_div(T *x, const T y, T *z, const size_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] / y;
    }
}

template <typename T>
class cuda_vector
{
private:
    T *m_data;
    size_t m_n; // data real length
public:
    cuda_vector() : m_data(nullptr), m_n(0)
    {
    }

    cuda_vector(const size_t n) : m_data(nullptr), m_n(n)
    {
        CHECK_ERROR(cudaMalloc(&m_data, n * sizeof(T)));
    }

    cuda_vector(cuda_vector &&other) : m_data(nullptr), m_n(0)
    {
        m_data = other.m_data;
        other.m_data = nullptr;
        m_n = other.m_n;
        other.m_n = 0;
    }

    cuda_vector(const cuda_vector &other) : m_data(nullptr), m_n(0)
    {
        m_n = other.m_n;
        CHECK_ERROR(cudaMalloc(&m_data, m_n * sizeof(T)));
        CHECK_ERROR(cudaMemcpy(m_data, other.m_data, m_n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    /*
     * @param [T*] host_data 主机端的数据
     * @param [const size_t] N 类型为T的数据的数量
     */
    cuda_vector(const T *host_data, const size_t N) : m_data(nullptr), m_n(0)
    {
        m_n = N;
        CHECK_ERROR(cudaMalloc(&m_data, N * sizeof(T)));
        CHECK_ERROR(cudaMemcpy(m_data, host_data, N * sizeof(T), cudaMemcpyHostToDevice));
    }

    cuda_vector &operator=(cuda_vector &&other)
    {
        if (m_data)
        {
            free();
        }
        m_data = other.m_data;
        other.m_data = nullptr;
        m_n = other.m_n;
        other.m_n = 0;
        return *this;
    }

    cuda_vector &operator=(const cuda_vector &other)
    {
        if (m_data && m_n != other.m_n)
        {
            free();
        }

        m_n = other.m_n;

        if (nullptr == m_data)
        {
            CHECK_ERROR(cudaMalloc(&m_data, other.size() * sizeof(T)));
        }

        CHECK_ERROR(cudaMemcpy(m_data, other.m_data, m_n * sizeof(T), cudaMemcpyDeviceToDevice));
        return *this;
    }

    inline void alloc(const size_t N)
    {
        if(m_data && N != m_n)
        {
            cudaFree(m_data);
            m_data = nullptr;
            m_n = 0;
        }
        m_n = N;
        CHECK_ERROR(cudaMalloc(m_data, N * sizeof(T)));
    }

    inline void free()
    {
        if (m_data)
        {
            CHECK_ERROR(cudaFree(m_data));
            m_data = nullptr;
            m_n = 0;
        }
    }

    ~cuda_vector()
    {
        free();
    }

    ///////

    T *data() const
    {
        return m_data;
    }

    size_t size() const
    {
        return m_n;
    }

    /////
    cuda_vector operator+(const cuda_vector &other)
    {
        if (this->m_n != other.m_n)
        {
            return {};
        }
        cuda_vector result(other.m_n);
        add(this->m_data, other.m_data, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator+=(const cuda_vector &other)
    {
        add(this->m_data, other.m_data, this->m_data, this->m_n);
        return *this;
    }

    cuda_vector operator-(const cuda_vector &other)
    {
        if (this->m_n != other.m_n)
        {
            return {};
        }
        cuda_vector result(other.m_n);
        sub(this->m_data, other.m_data, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator-=(const cuda_vector &other)
    {
        sub(this->m_data, other.m_data, this->m_data, this->m_n);
        return *this;
    }

    cuda_vector operator*(const cuda_vector &other)
    {
        if (this->m_n != other.m_n)
        {
            return {};
        }
        cuda_vector result(other.m_n);
        mul(this->m_data, other.m_data, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator*=(const cuda_vector &other)
    {
        mul(this->m_data, other.m_data, this->m_data, this->m_n);
        return *this;
    }

    cuda_vector operator/(const cuda_vector &other)
    {
        if (this->m_n != other.m_n)
        {
            return {};
        }
        cuda_vector result(other.m_n);
        div(this->m_data, other.m_data, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator/=(const cuda_vector &other)
    {
        div(this->m_data, other.m_data, this->m_data, this->m_n);
        return *this;
    }

    /////

    cuda_vector operator+(const T &other)
    {
        cuda_vector result(this->m_n);
        add(this->m_data, other, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator+=(const T &other)
    {
        add(this->m_data, other, this->m_data, this->m_n);
        return *this;
    }

    cuda_vector operator-(const T &other)
    {
        cuda_vector result(this->m_n);
        sub(this->m_data, other, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator-=(const T &other)
    {
        sub(this->m_data, other, this->m_data, this->m_n);
        return *this;
    }

    cuda_vector operator*(const T &other)
    {
        cuda_vector result(this->m_n);
        mul(this->m_data, other, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator*=(const T &other)
    {
        mul(this->m_data, other, this->m_data, this->m_n);
        return *this;
    }

    cuda_vector operator/(const T &other)
    {
        cuda_vector result(this->m_n);
        div(this->m_data, other, result.m_data, m_n);
        return std::move(result);
    }

    cuda_vector &operator/=(const T &other)
    {
        div(this->m_data, other, this->m_data, this->m_n);
        return *this;
    }

    static void add(T *x, const T *y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_add<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void sub(T *x, const T *y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_sub<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void mul(T *x, const T *y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_mul<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void div(T *x, const T *y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_div<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void add(T *x, const T y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_add<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void sub(T *x, const T y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_sub<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void mul(T *x, const T y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_mul<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }

    static void div(T *x, const T y, T *z, const size_t N)
    {
        const int block_size = 128;
        const int grid_size = (N + block_size - 1) / block_size;
        vector_div<<<grid_size, block_size>>>(x, y, z, N);
        return;
    }
};
