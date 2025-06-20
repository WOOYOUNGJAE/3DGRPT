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
        int depthLeft = static_cast<int>(params.customFloat3.y);
        float3 jitteredRayDir = rayDirectionFirst;

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


            if( payload.done || --depthLeft <= 0 ) // TODO RR, variable for depth
                break;

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
    float3 lightEmission = make_float3(params.customFloat3.x);
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
    next_render_pass = PGRNDTraceRTGaussiansPass;
    const float3 ray_o = pPayload->rayOri;       // Ray origin, when ray intersected the surface
    const float3 ray_d = pPayload->rayDir;    // Ray direction, when ray intersected the surface
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
        hitNormal = safe_normalize(pPayload->rayData->normal);
        // if (length(hitNormal) == 0.f)
        // {
        //     hitNormal = make_float3(0,0,1);
        // }
        hitRGB = volRadiance;
    }

    // Add object's emission color once
    if( pPayload->countEmitted )
        pPayload->emitted = EMISSION_COLOR;
    else
        pPayload->emitted = make_float3( 0.0f );

    // reset seed, ray_dir&pos from hemisphere sampling
    {
        const float z1 = rnd(rndSeed);
        const float z2 = rnd(rndSeed);

        float3 w_in;
        PT::cosine_sample_hemisphere( z1, z2, w_in );
        PT::Onb onb( hitNormal );
        onb.inverse_transform( w_in );
        pPayload->rayDir = w_in;
        pPayload->rayOri = ray_hitPos;
    }
    
    const float z1 = rnd(rndSeed);
    const float z2 = rnd(rndSeed);
    pPayload->rndSeed = rndSeed;

    float3 lightV1 = params.lightV1;
    float3 lightV2 = params.lightV2;
    float3 curLightPos = params.lightCorner + lightV1 * z1 + lightV2 * z2;


    float3 L = curLightPos - ray_hitPos;
    if (params.onOffFloat3.x == 1.f) // area Light
        L = curLightPos - ray_hitPos;
    else if (params.onOffFloat3.x == 0.f)
        L = params.lightCorner - ray_hitPos; // non-area light

    float occlusionRayMax = length(L);
    L = safe_normalize(L);
    const float nDl = dot( hitNormal, L );
    const float LnDl = 1.f; // TODO : -dot( light.normal, L );
    
    float weight = 0.0f;
    if ((!meshIsCloser) || (nDl > 0.f && LnDl > 0.f)) // ready to trace occlusion
    {
        // TRACE OCCLUSION
        // ray start pos
        float3 occlusion_ray_o = meshIsCloser ? 
        (ray_hitPos + L * TRACE_MESH_TMIN) : (ray_hitPos + L * EPS_SHIFT_GS);
        
        unsigned int is_occluded = traceOcclusion(
            occlusion_ray_o,
            L,
            occlusionRayMax - EPS_SHIFT_GS  // tmax
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
    pPayload->ptRadiance += lightEmission * weight;

    // -- Write outputs to pPayload --
    // Intersection point - also determines origin of next ray
    pPayload->t_hit = hit_t;
    // If ray has bounces remaining, update next ray orig and dir
    // Output: Number of times face redirected
    pPayload->numBounces = numBounces;
    
    if (pPayload->countEmitted && meshIsCloser == false)
    {
        if (weight != 0.f) weight = 1.f;
        // if (weight > 0.f) weight = 1.f;
        pPayload->ptRadiance = make_float3(params.customFloat3.z) * weight;

        if (params.onOffFloat3.y == 1.f)  // Use Secondary ray on gaussian
            pPayload->done = static_cast<unsigned int>((params.customFloat3.y - 1) == 0);
        else if (params.onOffFloat3.y == 0.f) // Use only Primary ray on Gaussian
            pPayload->done = 1;
    }

    pPayload->countEmitted = false;
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

