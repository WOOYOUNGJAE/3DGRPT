#pragma once

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};



namespace PT
{
    struct ParallelogramLight
    {
        float3 corner;
        float3 v1, v2;
        float3 normal;
        float3 emission;
    };
    
    static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
    {
        // Uniformly sample disk.
        const float r   = sqrtf( u1 );
        const float phi = 2.0f*M_PIf * u2;
        p.x = r * cosf( phi );
        p.y = r * sinf( phi );

        // Project up to hemisphere.
        p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
    }

    
    struct Onb
    {
        __forceinline__ __device__ Onb(const float3& normal)
        {
            m_normal = normal;

            if( fabs(m_normal.x) > fabs(m_normal.z) )
            {
            m_binormal.x = -m_normal.y;
            m_binormal.y =  m_normal.x;
            m_binormal.z =  0;
            }
            else
            {
            m_binormal.x =  0;
            m_binormal.y = -m_normal.z;
            m_binormal.z =  m_normal.y;
            }

            m_binormal = safe_normalize(m_binormal);
            m_tangent = cross( m_binormal, m_normal );
        }

        __forceinline__ __device__ void inverse_transform(float3& p) const
        {
            p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
        }

        float3 m_tangent;
        float3 m_binormal;
        float3 m_normal;
    };
}