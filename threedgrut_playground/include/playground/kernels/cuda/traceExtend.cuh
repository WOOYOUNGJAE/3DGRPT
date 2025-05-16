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

static __device__ __forceinline__ unsigned int traceOcclusion(const float3 rayOri, const float3 rayDir)
{
    unsigned int is_occluded;
    HybridRayPayload occlusionPayload;
    packPointer(&occlusionPayload, is_occluded);
    is_occluded = 0u;

    // check occluded by mesh
    optixTrace(
        params.triHandle,
        rayOri,
        rayDir,
        TRACE_MESH_TMIN,
        TRACE_MESH_TMAX, 0.0f,                // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1,                         // SBT offset
        1,                          // SBT stride
        0,                          // missSBTIndex,
        is_occluded
        );
    // check occluded by Gaussians
    optixTrace(
        params.handle,
        rayOri,
        rayDir,
        TRACE_MESH_TMIN,
        TRACE_MESH_TMAX, 0.0f,                // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1,                         // SBT offset
        1,                          // SBT stride
        0,                          // missSBTIndex,
        is_occluded
        );
    return is_occluded;
}

#endif