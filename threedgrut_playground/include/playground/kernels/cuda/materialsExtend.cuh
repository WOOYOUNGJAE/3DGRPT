
#pragma once
#ifdef __PLAYGROUND__MODE__

#include <playground/kernels/cuda/materials.cuh>



static __device__ __inline__ float3 shaded_gaussian(const float3 ray_d, float3 normal, const float3 radiance)
{
    float3 diffuse = radiance;

    float shade = fabsf(dot(ray_d, normal));
    return diffuse * shade;
}


#endif