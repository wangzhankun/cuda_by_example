#pragma once

#include <cstdio>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#define TICK(NAME) auto t1##NAME = std::chrono::steady_clock::now();
#define TOCK(NAME) std::cout << "["__FILE__"]" << ":" << __LINE__ << " "#NAME << " running time = " << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t1##NAME).count() << "ms." << std::endl;

#define ROUND_UP_DIV(x, base) ((x+base-1)/base)

#define GET_LAST_CUDA_ERROR()\
do{\
    cudaError_t err = cudaGetLastError();\
    printf("["__FILE__"]:%d CUDA ERROR: errno = %d, %s\n", __LINE__, err, cudaGetErrorString(err));\
}while(0);

#define CHECK_ERROR(call)\
do{\
    cudaError_t err = call;\
    if(err != cudaSuccess){\
        printf("%s:%d errno = %d, error = %s\n", "["__FILE__"]", __LINE__, err, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
}while(0);

#define CUDA_TICK(NAME) \
cudaEvent_t start##NAME, stop##NAME;\
CHECK_ERROR(cudaEventCreate(&start##NAME));\
CHECK_ERROR(cudaEventCreate(&stop##NAME));\
CHECK_ERROR(cudaEventRecord(start##NAME, 0));\
cudaEventQuery(start##NAME);

#define CUDA_TOCK(NAME) \
CHECK_ERROR(cudaEventRecord(stop##NAME, 0));\
CHECK_ERROR(cudaEventSynchronize(stop##NAME));\
float elapsedTime##NAME;\
CHECK_ERROR(cudaEventElapsedTime(&elapsedTime##NAME, start##NAME, stop##NAME));\
printf("%s:%d "#NAME" running time = %fms.\n", "["__FILE__"]", __LINE__, elapsedTime##NAME);\
CHECK_ERROR(cudaEventDestroy(start##NAME));\
CHECK_ERROR(cudaEventDestroy(stop##NAME));



///////////////////////user struct/////////////////

