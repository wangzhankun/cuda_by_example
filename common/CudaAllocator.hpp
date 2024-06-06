#include "global.hpp"

template <typename T>
struct CudaAllocator
{
    using value_type = T;
    T *allocate(size_t size)
    {
        T *p = nullptr;
        CHECK_ERROR(cudaMalloc(&p, size * sizeof(T)));
        return p;
    }
    void deallocate(T *p, size_t size = 0)
    {
        CHECK_ERROR(cudaFree(p));
    }
    template <typename... Args>
    void construct(T *p, Args &&...args)
    {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new ((void *)p) T(std::forward<Args>(args)...);
    }
};
