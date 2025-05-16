#pragma once
#ifdef __PLAYGROUND__MODE__

#include <playground/kernels/cuda/trace.cuh>

/**
 * @overload : pack single pointer to payload
 */
static __device__ __forceinline__ void packPointer(void* ptr, unsigned int& i0)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
}

#endif