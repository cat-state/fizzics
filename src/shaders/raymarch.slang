// Simple HLSL Raymarcher
readonly uint2 resolution;

RWStructuredBuffer<float4> render_output;

float sdf(float3 p) {
    return length(p) - 0.5;
}

float3 estimate_normal(float3 p) {
    const float eps = 0.001;
    const float2 h = float2(eps, -eps);
    return normalize(
        h.xyy * sdf(p + h.xyy) +
        h.yxy * sdf(p + h.yxy) +
        h.yyx * sdf(p + h.yyx) +
        h.xxx * sdf(p + h.xxx)
    );
}

float4 raymarch(float3 ro, float3 rd) {
    float t = 0.0;
    for (int i = 0; i < 100; i++) {
        float d = sdf(ro);
        if (d < 0.001 * (t + 1.0)) {
            return float4(ro, 1.0);
        } else if (t > 100.0) break;
        t += d;
        ro += rd * d;
    }
    return float4(0, 0, 0, 0);
}

// Main entry point
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    float2 uv = float2(id.xy) / float(resolution.y);
    uv = uv * 2.0 - 1.0;

    float3 ro = float3(0, 0, -1);
    float3 rd = normalize(float3(uv, 1));
    float4 color = raymarch(ro, rd);
    if (color.w > 0.0) {
        color.xyz = estimate_normal(color.xyz);
        color.xyz = color.xyz * 0.5 + 0.5;
    }
    uint2 pixel_pos = id.xy;
    uint pixel_index = pixel_pos.x + pixel_pos.y * resolution.x;
    render_output[pixel_index] = color;
}
