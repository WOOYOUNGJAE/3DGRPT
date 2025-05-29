// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#define __PLAYGROUND__MODE__ 1

#include <playground/pipelineParameters.h>

extern "C"
{
    #ifndef __PLAYGROUND__PARAMS__
    __constant__ PlaygroundPipelineParameters params;
    #define __PLAYGROUND__PARAMS__ 1
    #endif
}

#include <optix.h>
#include <playground/kernels/cuda/traceExtend.cuh>
#include <playground/kernels/cuda/materialsExtend.cuh>

#define USE_SHADOW 1
#define USE_GS_SHADING 0
__constant__ unsigned int SAMPLES_PER_LAUNCH = 1;
extern "C" __global__ void __raygen__rg() {

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 rayOriginFirst    = params.rayWorldOrigin(idx);
    const float3 rayDirectionFirst = params.rayWorldDirection(idx);
    // Ray coordinates in pixels
    const int rx = fminf(idx.x, params.frameBounds.x);
    const int ry = fminf(idx.y, params.frameBounds.y);

    // jitter
    unsigned int seed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);
    const int width   = params.width;
    const int height   = params.height;

    float3 U = FLOAT4_TO_FLOAT3(params.rayToWorld[0]);
    float3 V = FLOAT4_TO_FLOAT3(params.rayToWorld[1]);
    float3 W = FLOAT4_TO_FLOAT3(params.rayToWorld[2]);
    
    float3 resultRGB = make_float3( 0.0f );
    int remaningSamples = SAMPLES_PER_LAUNCH;
    PT::RayPayload payload;
    do
    {
        // Initialize Payload
        payload.initialize();
        payload.rndSeed = seed;
        const float ray_t_max = params.rayMaxT[idx.z][ry][rx];
        RayData rayData;
        rayData.initialize();
        payload.rayData = &rayData;
        payload.rayOri = rayOriginFirst;
        int depth = 0;
        float3 jitteredRayDir = rayDirectionFirst;

        // const float2 subpixel_jitter = make_float2( rnd( seed )-0.5f, rnd( seed )-0.5f );
        // const float2 d = 2.0f * make_float2(
        //     ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( width ), //width
        //     ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( height ) // height
        //     ) - 1.0f;     
        //     jitteredRayDir = safe_normalize(rayDirectionFirst 
        //     + d.x * U
        //     + d.y * V);
        payload.rayDir = jitteredRayDir;

        float3 rayOrigin;
        for( ;; )
        {
            PT::traceRadiance(
                0.01f,  // tmin       // TODO: smarter offset
                ray_t_max,  // tmax
                &payload );

            resultRGB += payload.emitted;
            resultRGB += payload.ptRadiance * payload.attenuationRGB;
                    
            if( payload.done  || depth >= 3 ) // TODO RR, variable for depth
                break;
                
            ++depth;
        }
    } while (--remaningSamples);
    
    
    
    // Write back to global mem in launch params
    float4 rgba = make_float4(resultRGB.x / static_cast<float>(SAMPLES_PER_LAUNCH),
                              resultRGB.y / static_cast<float>(SAMPLES_PER_LAUNCH),
                              resultRGB.z / static_cast<float>(SAMPLES_PER_LAUNCH),
                                    payload.accumulatedAlpha);

    writeRadianceDensityToOutputBuffer(rgba);
    writeUpdatedRaysToBuffer(payload.lastRayOri, payload.lastRayDir);
}

static __device__ __inline__ float3 getSmoothNormal()
{
    // Uses interpolated vertex normals to get a smooth varying interpolated normal
    const unsigned int triId = optixGetPrimitiveIndex();
    const unsigned int v0_idx = params.triangles[triId][0];
    const unsigned int v1_idx = params.triangles[triId][1];
    const unsigned int v2_idx = params.triangles[triId][2];
    const float3 n0 = make_float3(params.vNormals[v0_idx][0], params.vNormals[v0_idx][1], params.vNormals[v0_idx][2]);
    const float3 n1 = make_float3(params.vNormals[v1_idx][0], params.vNormals[v1_idx][1], params.vNormals[v1_idx][2]);
    const float3 n2 = make_float3(params.vNormals[v2_idx][0], params.vNormals[v2_idx][1], params.vNormals[v2_idx][2]);
    const float2 barycentric = optixGetTriangleBarycentrics();
    float3 interpolated_normal = (1 - barycentric.x - barycentric.y) * n0 + barycentric.x * n1 + barycentric.y * n2;
    interpolated_normal /= length(interpolated_normal);

    return interpolated_normal;
}

static __device__ __inline__ float3 getHardNormal()
{
    // Computes a "hard" non-varying normal using the vertex positions
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int triId = optixGetPrimitiveIndex();
    const unsigned int gasSbtIdx = optixGetSbtGASIndex();
    float3 v[3] = {};
    optixGetTriangleVertexData(gas, triId, gasSbtIdx, 0, v);
    float3 normal = safe_normalize(cross(v[1] - v[0], v[2] - v[0]));
    return normal;
}

extern "C" __global__ void __closesthit__occlusion__ch()
{
    optixSetPayload_0( static_cast<unsigned int>( 1 ) );
    optixSetPayload_1( __float_as_uint( optixGetRayTmax() ) ); // dummy for temp
}

extern "C" __global__ void __closesthit__ch()
{
    // Only for MESH

    // Read inputs off payload
    PT::RayPayload* pPayload = PT::getRayPayload<PT::RayPayload>();
    // get payload value
    unsigned int numBounces = pPayload->numBounces;  // Number of times ray was reflected so far
    unsigned int next_render_pass = getNextTraceState();
    unsigned int rndSeed = pPayload->rndSeed;

    // get triangle info
    const unsigned int triId = optixGetPrimitiveIndex();
    // Compute normals using interplated precomputed vertex normals or directly from vertex positions ("non-smooth")
    float3 hitNormal = (params.playgroundOpts & PGRNDRenderSmoothNormals) ? getSmoothNormal() : getHardNormal();
    auto intersected_type = params.primType[triId][0];
    const float3 triDiffuse = get_pure_diffuse();

    // Ready for tracing Gaussians
    float3 new_ray_dir = make_float3(0.0, 0.0, 0.0);
    next_render_pass = PGRNDTraceRTGaussiansPass;
    const float3 ray_o = optixGetWorldRayOrigin();       // Ray origin, when ray intersected the surface
    const float3 ray_d = optixGetWorldRayDirection();    // Ray direction, when ray intersected the surface
    float hit_t = optixGetRayTmax();                     // t when ray intersected the surface

    // TODO Later: Assumed intersected_type == PGRNDPrimitiveDiffuse
    // Trace Gaussian
    float gaussianClosestHit_t = TRACE_MAX;
    const float4 volumetricRadDns = traceGaussians_outDist<PT::RayPayload>(*(pPayload->rayData), ray_o, ray_d, 1e-9, hit_t, pPayload, gaussianClosestHit_t/*out*/);
    float3 volRadiance = make_float3(volumetricRadDns.x, volumetricRadDns.y, volumetricRadDns.z);
    const float volAlpha = volumetricRadDns.w;
    
    unsigned int meshIsCloser = (hit_t < gaussianClosestHit_t) | (pPayload->rayData->hitCount == 0.f/*Miss Gaussian*/);

    float3 ray_hitPos;
    float3 hitRGB; // mesh diffuse or volRadiance
    if (meshIsCloser)
    {
        ray_hitPos = ray_o + hit_t * ray_d;
        hitRGB = triDiffuse; // Apply diffuse to attenuation
    }
    else // Gaussian is Closer
    {
        ray_hitPos = ray_o + gaussianClosestHit_t * ray_d;
        hitNormal = pPayload->rayData->normal;
        hitRGB = volRadiance; // Apply volRadiance(as diffuse of gaussian) to attenuation
    }

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

    float3 lightV1 = params.lightV1;
    float3 lightV2 = params.lightV2;
    float3 curLightPos = params.lightCorner + lightV1 * z1 + lightV2 * z2;


    float3 L = curLightPos - ray_hitPos;
    float occlusionRayMax = length(L);
    L = safe_normalize(L);
    const float nDl = dot( hitNormal, L );
    const float LnDl = 1.f; // TODO : -dot( light.normal, L );
    
    float weight = 0.0f;
    if (nDl > 0.f && LnDl > 0.f) // ready to trace occlusion
    {
        // TRACE OCCLUSION
        // ray start pos
        float3 occlusion_ray_o = meshIsCloser ? 
        (ray_hitPos + hitNormal * TRACE_MESH_TMIN) : (ray_hitPos + L * EPS_SHIFT_GS);
        
        unsigned int is_occluded = traceOcclusion(
            occlusion_ray_o,
            L,
            occlusionRayMax - 0.01f  // tmax
            );

        if( !is_occluded )
        {
            const float A = length(cross(lightV1, lightV2));
            weight = nDl * LnDl * A / (M_PIf * occlusionRayMax * occlusionRayMax);
        }
    }

    // process color info


    pPayload->accumulatedAlpha += 1.f; // Assume this is diffuse
    pPayload->attenuationRGB *= hitRGB;
    pPayload->ptRadiance += params.lightEmission * weight;

    // -- Write outputs to pPayload --
    // Intersection point - also determines origin of next ray
    pPayload->t_hit = hit_t;
    // If ray has bounces remaining, update next ray orig and dir
    pPayload->rayOri = ray_o + hit_t * ray_d;
    pPayload->rayDir = new_ray_dir;
    // Output: Number of times face redirected
    pPayload->numBounces = numBounces;
    // Update next seed if RNG was used
    pPayload->rndSeed = rndSeed;

    // Output: Ray hit something so it is considered redirected (->Gaussians pass), or terminate
    setNextTraceState(PGRNDTraceTerminate);
}

extern "C" __global__ void __intersection__is() {
    intersectVolumetricGS();
}

extern "C" __global__ void __anyhit__ah()
{
    // Enabled only for gaussian ray tracing
    if (getNextTraceState() == PGRNDTraceRTGaussiansPass)
        anyhitSortVolumetricGS();
}

extern "C" __global__ void __miss__ms()
{
    // Ray missed: no primitives - trace remaining gaussians till bbox end, or terminate
    if (getNextTraceState() == PGRNDTracePrimitivesPass)
    {
        PT::RayPayload* pPayload = PT::getRayPayload<PT::RayPayload>();

        pPayload->rayMissed = true;
        pPayload->rayOri = make_float3(0.0);
        pPayload->rayDir = make_float3(0.0);
        setNextTraceState(PGRNDTraceRTLastGaussiansPass);
    }
}


extern "C" __global__ void __miss__occlusion__ms()
{

}

#undef __PLAYGROUND__MODE__

