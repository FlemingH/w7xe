#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "[HIP ERROR] %s:%d  %s\n  %s\n",             \
                    __FILE__, __LINE__, hipGetErrorName(err),              \
                    hipGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
