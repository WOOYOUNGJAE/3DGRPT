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

/**
 * @brief Ray Payload for PathTracing
 * @note accumulated color(radiance) is in pHybridPRD->accumulatedColor
 */
struct PTRayPayload {
    float3       emitted;
    float3       attenuation;
    float3       origin;
    float3       direction;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;

    HybridRayPayload* pHybridPRD; // per Ray Data for Mesh-GS Tracing


    __device__ __forceinline__ void initialize() {
        emitted      = make_float3(0.f);
        attenuation  = make_float3(1.f);
        countEmitted = true;
        done         = false;
        seed         = 0u;
    }
};

/**
 * @return closest distance
 */
static __device__ __inline__ float traceVolumetricGS_outDist(
    RayData& rayData,
    const float3& rayOrigin,
    const float3& rayDirection,
    const float tmin,
    const float tmax) {
    bool isFirstLoop = true;
    float outDistance = -1.f;
    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return -1.f;
    }

    float rayTransmittance = 1.0f - rayData.density;
    float2 minMaxT       = intersectAABB(params.aabb, rayOrigin, rayDirection);
    minMaxT.x = fmaxf(minMaxT.x, tmin);
    minMaxT.y = fminf(minMaxT.y, tmax);
    constexpr float epsT = 1e-9;

    float rayLastHitDistance = fmaxf(0.0f, minMaxT.x - epsT);
    RayPayload rayPayload;

    while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance)) {
        trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
        if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = rayPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > params.minTransmittance)) {
                const float hitWeight = particleDensityProcessHitFwdFromBuffer(
                    rayOrigin,
                    rayDirection,
                    rayHit.particleId,
                    {{(gaussianParticle_RawParameters_0*)params.particleDensity, nullptr}},
                    &rayTransmittance,
                    &rayData.hitDistance,
#ifdef ENABLE_NORMALS
                    true, &rayData.normal
#else
                    false, nullptr
#endif
                );

                particleFeaturesIntegrateFwdFromBuffer(rayDirection,
                                                       hitWeight,
                                                       rayHit.particleId,
                                                       {{(float3*)params.particleRadiance, nullptr}, params.sphDegree},
                                                       &rayData.radiance);

                rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);

#ifdef ENABLE_HIT_COUNTS
                rayData.hitCount += hitWeight > 0.f ? 1.0f : 0.f;
#endif
                if (isFirstLoop == true)// // First Loop
                {
                    isFirstLoop = false;
                    outDistance = rayHit.distance;
                }
            }
        }
    }

    rayData.density = 1 - rayTransmittance;
    rayData.rayLastHitDistance = rayLastHitDistance;
    return outDistance;
}


static __device__ __forceinline__ float4 traceGaussians_outDist(
    RayData& rayData,
    const float3& rayOrigin,
    const float3& rayDirection,
    const float tmin,
    const float tmax,
    HybridRayPayload* payload,
    float& outClosestDistance) {

   const uint3 idx = optixGetLaunchIndex();
   const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
   const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

   if (params.playgroundOpts & PGRNDRenderDisableGaussianTracing)
       return make_float4(0.0);

   // Copy RayData, to avoid writing the output buffer by this pass
   RayData prevRayData = rayData;
   setNextTraceState(PGRNDTraceRTGaussiansPass);
   outClosestDistance = traceVolumetricGS_outDist(rayData, rayOrigin, rayDirection, tmin, tmax);

   // The difference in the output buffer is the result of this trace path
   float4 accumulated_radiance = make_float4(
        rayData.radiance.x - prevRayData.radiance.x,
        rayData.radiance.y - prevRayData.radiance.y,
        rayData.radiance.z - prevRayData.radiance.z,
        rayData.density - prevRayData.density
   );

    payload->lastRayOri = rayOrigin;
    payload->lastRayDir = rayDirection;

   return accumulated_radiance;
}

/**
 * @return if Occluded
 */
static __device__ __inline__ bool traceOcclusion_GS(
    RayData& rayData,
    const float3& rayOrigin,
    const float3& rayDirection,
    const float tmin,
    const float tmax) {

    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return false;
    }

    float rayTransmittance = 1.0f - rayData.density;
    float2 minMaxT       = intersectAABB(params.aabb, rayOrigin, rayDirection);
    minMaxT.x = fmaxf(minMaxT.x, tmin);
    minMaxT.y = fminf(minMaxT.y, tmax);
    constexpr float epsT = 1e-9;

    float rayLastHitDistance = fmaxf(0.0f, minMaxT.x - epsT);
    RayPayload rayPayload;
    float minTransmittance = 0.4f;
    unsigned int timeout = 0;
    trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
        if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
            return false;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = rayPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) /*&& (rayTransmittance > minTransmittance)*/) {
                const float hitWeight = particleDensityProcessHitFwdFromBuffer(
                    rayOrigin,
                    rayDirection,
                    rayHit.particleId,
                    {{(gaussianParticle_RawParameters_0*)params.particleDensity, nullptr}},
                    &rayTransmittance,
                    &rayData.hitDistance,
#ifdef ENABLE_NORMALS
                    true, &rayData.normal
#else
                    false, nullptr
#endif
                );

                particleFeaturesIntegrateFwdFromBuffer(rayDirection,
                                                       hitWeight,
                                                       rayHit.particleId,
                                                       {{(float3*)params.particleRadiance, nullptr}, params.sphDegree},
                                                       &rayData.radiance);

                rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);

#ifdef ENABLE_HIT_COUNTS
                rayData.hitCount += hitWeight > 0.f ? 1.0f : 0.f;
#endif
            }
        }
//     while ((rayLastHitDistance <= minMaxT.y) /*&& (rayTransmittance > params.minTransmittance)*/) {
//         trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
//         if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
//             break;
//         }

// #pragma unroll
//         for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
//             const RayHit rayHit = rayPayload[i];

//             if ((rayHit.particleId != RayHit::InvalidParticleId) /*&& (rayTransmittance > minTransmittance)*/) {
//                 const float hitWeight = particleDensityProcessHitFwdFromBuffer(
//                     rayOrigin,
//                     rayDirection,
//                     rayHit.particleId,
//                     {{(gaussianParticle_RawParameters_0*)params.particleDensity, nullptr}},
//                     &rayTransmittance,
//                     &rayData.hitDistance,
// #ifdef ENABLE_NORMALS
//                     true, &rayData.normal
// #else
//                     false, nullptr
// #endif
//                 );

//                 particleFeaturesIntegrateFwdFromBuffer(rayDirection,
//                                                        hitWeight,
//                                                        rayHit.particleId,
//                                                        {{(float3*)params.particleRadiance, nullptr}, params.sphDegree},
//                                                        &rayData.radiance);

//                 rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);

// #ifdef ENABLE_HIT_COUNTS
//                 rayData.hitCount += hitWeight > 0.f ? 1.0f : 0.f;
// #endif
//             }
//         }
//         timeout += 1;
//         if (timeout > 1)
//             break;
//     }

    rayData.density = 1 - rayTransmittance;
    rayData.rayLastHitDistance = rayLastHitDistance;

    return rayTransmittance < minTransmittance;
}


static __device__ __forceinline__ unsigned int traceOcclusion(const float3 rayOri, const float3 rayDir, const float rayMax)
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
        rayMax, 0.0f,                // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1,                         // SBT offset
        1,                          // SBT stride
        0,                          // missSBTIndex,
        is_occluded
        );

    RayData rayData;
    rayData.initialize();
    occlusionPayload.rayData = &rayData;
    // check occluded by Gaussians
    optixTrace(
        params.handle,
        rayOri,
        rayDir,
        TRACE_MESH_TMIN,
        rayMax, 0.0f,                // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1,                         // SBT offset
        1,                          // SBT stride
        0,                          // missSBTIndex,
        is_occluded
        );
    
    // is_occluded |= traceOcclusion_GS(rayData, rayOri, rayDir, TRACE_MESH_TMIN, TRACE_MESH_TMAX);

    
    return is_occluded;
}

#endif