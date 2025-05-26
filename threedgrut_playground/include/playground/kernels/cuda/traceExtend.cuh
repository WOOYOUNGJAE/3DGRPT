#pragma once
#ifdef __PLAYGROUND__MODE__

#include <playground/kernels/cuda/trace.cuh>
#include <playground/kernels/cuda/rng.cuh>
#include <playground/pathTracing.cuh>

constexpr uint32_t MAX_BOUNCES = 32;           // Maximum number of mirror material bounces only (irrelevant to pbr)
constexpr uint32_t TIMEOUT_ITERATIONS = 1000;  // Terminate ray after max iterations to avoid infinite loop
constexpr float REFRACTION_EPS_SHIFT = 1e-5;   // Add eps amount to refracted rays pos to avoid repeated collisions

constexpr float EPS_SHIFT_GS = 0.1f; // Add eps amount to secondary rays pos to avoid repeated collisions for Gaussian Tracing
constexpr float TRACE_MAX = 1e5;
__constant__ float3 LIGHT_POS = {0.0f, -6.0f, 0.0f}; // only for point light
__constant__ float3 LIGHT_CORNER = {-4.0f, 2.245f, 9.f}; // cornell box : -4, 2.245, 3.78
__constant__ float3 LIGHT_V1 = {2.0f, 0.0f, -2.0f};
__constant__ float3 LIGHT_V2 = {2.0f, 0.0f, 2.0f};
__constant__ float3 LIGHT_NORMAL = {0.0f, 0.0f, -1.0f};
__constant__ float3 LIGHT_EMISSION = {200.f, 200.f, 200.f}; // Light Color
__constant__ float3 EMISSION_COLOR = {0.f, 0.f, 0.f}; // {15.f, 15.f, 5.f}; // If Emission Object: this, Non-Emmision Object: Zero
/**
 * @overload : pack single pointer to payload
 */
static __device__ __forceinline__ void packPointer(void* ptr, unsigned int& i0)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
}


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


template <typename T>
static __device__ __forceinline__ float4 traceGaussians_outDist(
    RayData& rayData,
    const float3& rayOrigin,
    const float3& rayDirection,
    const float tmin,
    const float tmax,
    T* payload,
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

namespace PT
{
    /**
     * @brief Ray Payload for PathTracing
     */
    struct RayPayload {
        float3       emitted;           // light emitted, only for first hit
        float3       attenuationRGB;
        float3       rayOri;             // next ray origin to use
        float3       rayDir;             // next ray dir to use, if ray was reflected, refracted, etc

        // PBR params
        float3 ptRadiance;         // 
        float3 lastRayOri;          // Last ray origin used to trace gaussians
        float3 lastRayDir;          // Last ray direction used to trace gaussians
        float  accumulatedAlpha;    // Amount of density accumulated by the ray so far. Solid mesh faces count as opaque.
        float  blockingRadiance;     // Total radiance accumulated only by volumetric radiance integration so far
        unsigned int rndSeed;      // random seed for current ray
        int          countEmitted; // True: apply direct light. False: apply only indirect light
        
        int          done;
        float        t_hit;               // ray t of latest intersection
        unsigned int numBounces;   // current number of reflectance bounces
        bool rayMissed;             // True if ray missed

        RayData* rayData;
    
        __device__ __forceinline__ void initialize() {
            emitted      = make_float3(0.f);
            attenuationRGB  = make_float3(1.f);
            ptRadiance = make_float3(0.f);
            accumulatedAlpha = 0.f;
            blockingRadiance = 0.f;
            rndSeed = 0u;
            countEmitted = true;
            done         = false;
            t_hit = 0.f;
            numBounces = 0u;
            rayMissed = false;
            // seed         = 0u;
        }
    };

    template <typename T>
    static __device__ /* TODO Later __forceinline__*/ T* getRayPayload()
    {
        const unsigned int u0 = optixGetPayload_0();
        const unsigned int u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }



    static __device__ __forceinline__ void traceRadiance_Mesh(const float3 rayOri, const float3 rayDir, PT::RayPayload* pPayload)
    {
        setNextTraceState(PGRNDTracePrimitivesPass);
    
        unsigned int p0, p1;
        packPointer(pPayload, p0, p1);
        optixTrace(
            params.triHandle,
            rayOri,
            rayDir,
            TRACE_MESH_TMIN,          // Min intersection distance
            TRACE_MESH_TMAX,          // Max intersection distance
            0.0f,                     // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, // | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            0, // SBT offset   -- See SBT discussion
            1, // SBT stride   -- See SBT discussion
            0, // missSBTIndex -- See SBT discussion
            p0, p1
        );
    }
    
    /**
     * @brief for PathTracing for now. original RT pipeline(trace mesh->gaussian) in here
     */
    static __device__ __forceinline__ unsigned int traceRadiance(const float3 rayOri, const float3 rayDir, const float rayMin, const float rayMax, PT::RayPayload* pPayload)
    {
        unsigned int timeout = 0;
    
        // Termination criteria:
        // 1. Ray missed surface (ray dir is 0), or
        // 2. PBR Materials: No remaining bounces, or
        // 3. Mirrors: No remaining bounces
        // while ((length(pPayload->rayDir) > 0.1) &&
        //        (pPayload->pbrNumBounces < params.maxPBRBounces) && (pPayload->numBounces < MAX_BOUNCES))
        while (true)
        {
            float3 rayOri = pPayload->rayOri;
            float3 rayDir = pPayload->rayDir;
            // Process ClosestHit, AnyHit, Miss for Mesh
            traceRadiance_Mesh(rayOri, rayDir, pPayload);
    
            // Ratio of the light which didn't go through the material: [0,1], where 1.0 means no light went through
            // (TODO: This is actually the inverse transmittance)
    
            if (getNextTraceState() == PGRNDTraceTerminate)
                break;
    
            // Invoke 3drt shader to integrate all gaussians until the surface is hit
            float4 volumetricRadDns;
            if (getNextTraceState() == PGRNDTraceRTLastGaussiansPass)  // Process After Miss Mesh
            {
                // Trace Gaussian
                float gaussianClosestHit_t = TRACE_MAX;
                volumetricRadDns = traceGaussians_outDist(*(pPayload->rayData), rayOri, rayDir, 1e-9, rayMax, pPayload, gaussianClosestHit_t/*out*/);
                float3 volRadiance = make_float3(volumetricRadDns.x, volumetricRadDns.y, volumetricRadDns.z);
                const float volAlpha = volumetricRadDns.w;
                unsigned int hit = pPayload->rayData->hitCount > 0;

                float3 ray_hitPos = rayOri + gaussianClosestHit_t * rayDir;
                float3 hitNormal = pPayload->rayData->normal;
                
                // Add object's emission color once
                if( pPayload->countEmitted )
                    pPayload->emitted = EMISSION_COLOR;
                else
                    pPayload->emitted = make_float3( 0.0f );

                // reset seed, ray_dir&pos from hemisphere sampling
                unsigned int seed = pPayload->rndSeed;
                {
                    const float z1 = rnd(seed);
                    const float z2 = rnd(seed);

                    float3 w_in;
                    PT::cosine_sample_hemisphere( z1, z2, w_in );
                    PT::Onb onb( hitNormal );
                    onb.inverse_transform( w_in );
                    pPayload->rayDir = w_in;
                    pPayload->rayOri = ray_hitPos;

                    pPayload->countEmitted = false;
                }
                
                const float z1 = rnd(seed);
                const float z2 = rnd(seed);
                pPayload->rndSeed = seed;

                float3 curLightPos = LIGHT_CORNER + LIGHT_V1 * z1 + LIGHT_V2 * z2;

                float3 L = curLightPos - ray_hitPos;
                float occlusionRayMax = length(L);
                L = safe_normalize(L);
                const float nDl = dot( hitNormal, L );
                const float LnDl = 1.f; // TODO : -dot( light.normal, L );
                
                float weight = 0.0f;
                if (hit && nDl > 0.f && LnDl > 0.f) // ready to trace occlusion
                {
                    // TRACE OCCLUSION
                    // ray start pos
                    float3 occlusion_ray_o = ray_hitPos + L * EPS_SHIFT_GS;
                    
                    unsigned int is_occluded = traceOcclusion(
                        occlusion_ray_o,
                        L,
                        occlusionRayMax - 0.01f  // tmax
                        );

                    if( !is_occluded )
                    {
                        const float A = length(cross(LIGHT_V1, LIGHT_V2));
                        weight = nDl * LnDl * A / (M_PIf * occlusionRayMax * occlusionRayMax);
                    }
                }
                
                pPayload->attenuationRGB *= (volRadiance * volAlpha); // Apply volRadiance(as diffuse of gaussian) to attenuation
                pPayload->ptRadiance += LIGHT_EMISSION * weight;
                pPayload->accumulatedAlpha += volAlpha;

                setNextTraceState(PGRNDTraceTerminate);
            }
            // else // TODO : for non-diffuse mesh pass
            // {
            //     // Gaussians between PBR surfaces are integrated as volumetric radiance that directly contributes
            //     // to the final ray color
            //     volumetricRadDns = traceGaussians(rayData, rayOri, rayDir, 1e-9, payload.t_hit, &payload);
            //     float3 radiance = make_float3(volumetricRadDns.x, volumetricRadDns.y, volumetricRadDns.z);
            //     float alpha = volumetricRadDns.w;
            //     payload.accumulatedColor += make_float3(1.0f - payload.accumulatedAlpha) * radiance;
            //     payload.accumulatedAlpha = clamp(payload.accumulatedAlpha + alpha + payload.lastPBRTransmittance, 0.0f, 1.0f);
            //     payload.directLight += clampf3(radiance * alpha, 0.0f, 1.0f);
            //     payload.blockingRadiance = clamp(payload.blockingRadiance + alpha, 0.0f, 1.0f);
            // }
            timeout += 1;
            if (timeout > TIMEOUT_ITERATIONS)
                break;

            break;
        }
        pPayload->countEmitted = false;
        // payload.accumulatedColor += payload.directLight * (1.0f - payload.blockingRadiance);
    }
}

#endif