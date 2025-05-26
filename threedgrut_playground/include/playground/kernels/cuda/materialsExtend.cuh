
#pragma once
#ifdef __PLAYGROUND__MODE__

#include <playground/kernels/cuda/materials.cuh>


static __device__ __inline__ float3 get_pure_diffuse()
{
    const unsigned int triId = optixGetPrimitiveIndex();
    const unsigned int materialId = params.matID[triId][0];
    const auto material = get_material(materialId);

    const float2 uv0 = make_float2(params.matUV[triId][0][0], params.matUV[triId][0][1]);
    const float2 uv1 = make_float2(params.matUV[triId][1][0], params.matUV[triId][1][1]);
    const float2 uv2 = make_float2(params.matUV[triId][2][0], params.matUV[triId][2][1]);
    const float2 barycentric = optixGetTriangleBarycentrics();
    float2 texCoords = (1 - barycentric.x - barycentric.y) * uv0 + barycentric.x * uv1 + barycentric.y * uv2;

    float3 diffuse;
    bool disableTextures = params.playgroundOpts & PGRNDRenderDisablePBRTextures;

    float3 diffuseFactor = make_float3(material.diffuseFactor.x, material.diffuseFactor.y, material.diffuseFactor.z);
    if (!material.useDiffuseTexture || disableTextures)
    {
        diffuse = diffuseFactor;
    }
    else
    {
        cudaTextureObject_t diffuseTex = material.diffuseTexture;
        float4 diffuse_fp4 = tex2D<float4>(diffuseTex, texCoords.x, texCoords.y);
        diffuse = make_float3(diffuse_fp4.x, diffuse_fp4.y, diffuse_fp4.z);
        diffuse *= diffuseFactor;
    }

    return diffuse;
}

static __device__ __inline__ float3 shaded_gaussian(const float3 ray_d, float3 normal, const float3 radiance)
{
    float3 diffuse = radiance;

    float shade = fabsf(dot(ray_d, normal));
    return diffuse * shade;
}


#endif