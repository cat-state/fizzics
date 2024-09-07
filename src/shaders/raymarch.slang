// Simple HLSL Raymarcher
readonly uint2 resolution;

RWStructuredBuffer<float4> render_output;

// Define a distance field for a sphere
float sdSphere(float3 p, float radius)
{
    return length(p) - radius;
}

// Define the scene
float map(float3 p)
{
    return sdSphere(p, 1.0);
}

// Calculate normal
float3 calcNormal(float3 p)
{
    const float eps = 0.001;
    const float2 h = float2(eps, 0);
    return normalize(float3(
        map(p + h.xyy) - map(p - h.xyy),
        map(p + h.yxy) - map(p - h.yxy),
        map(p + h.yyx) - map(p - h.yyx)
    ));
}

// Raymarching function
float4 raymarch(float3 ro, float3 rd)
{
    float t = 0.0;
    for (int i = 0; i < 100; i++)
    {
        float3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001)
        {
            float3 normal = calcNormal(p);
            float3 color = normal * 0.5 + 0.5;
            return float4(color, 1.0);
        }
        if (t > 100.0) break;
        t += d;
    }
    return float4(0, 0, 0, 0);
}

// Main entry point
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float2 uv = float2(id.xy) / float(resolution.y);
    uv = uv * 2.0 - 1.0;

    float3 ro = float3(0, 0, -3);
    float3 rd = normalize(float3(uv, 1));
    float4 color = raymarch(ro, rd);
    uint2 pixel_pos = id.xy;
    uint pixel_index = pixel_pos.x + pixel_pos.y * resolution.x;
    render_output[pixel_index] = color;
}
