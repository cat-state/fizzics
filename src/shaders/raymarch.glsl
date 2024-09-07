#version 450
layout(row_major) uniform;
layout(row_major) buffer;

#line 4 0
layout(std430, binding = 1) buffer StructuredBuffer_vectorx3Cfloatx2C4x3E_t_0 {
    vec4 _data[];
} render_output_0;

#line 78 1
struct GlobalParams_std140_0
{
    readonly uvec2 resolution_0;
};


#line 78
layout(binding = 0)
layout(std140) uniform _S1
{
    readonly uvec2 resolution_0;
}globalParams_0;

#line 7 0
float sdSphere_0(vec3 p_0, float radius_0)
{
    return length(p_0) - radius_0;
}


float map_0(vec3 p_1)
{
    return sdSphere_0(p_1, 1.0);
}


vec3 calcNormal_0(vec3 p_2)
{

#line 19
    const vec3 _S2 = vec3(0.00100000004749745, 0.0, 0.0);

#line 19
    const vec3 _S3 = vec3(0.0, 0.00100000004749745, 0.0);

#line 19
    const vec3 _S4 = vec3(0.0, 0.0, 0.00100000004749745);



    return normalize(vec3(map_0(p_2 + _S2) - map_0(p_2 - _S2), map_0(p_2 + _S3) - map_0(p_2 - _S3), map_0(p_2 + _S4) - map_0(p_2 - _S4)));
}


#line 31
vec4 raymarch_0(vec3 ro_0, vec3 rd_0)
{

#line 31
    int i_0 = 0;

#line 31
    float t_0 = 0.0;

#line 47
    const vec4 _S5 = vec4(0.0, 0.0, 0.0, 0.0);

#line 47
    for(;;)
    {

#line 34
        if(i_0 < 100)
        {
        }
        else
        {

#line 34
            break;
        }
        vec3 p_3 = ro_0 + rd_0 * t_0;
        float d_0 = map_0(p_3);
        if(d_0 < 0.00100000004749745)
        {


            return vec4(calcNormal_0(p_3) * 0.5 + 0.5, 1.0);
        }
        if(t_0 > 100.0)
        {

#line 44
            break;
        }

#line 45
        float t_1 = t_0 + d_0;

#line 34
        i_0 = i_0 + 1;

#line 34
        t_0 = t_1;

#line 34
    }

#line 47
    return _S5;
}




layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 55
    uvec2 _S6 = gl_GlobalInvocationID.xy;

#line 61
    render_output_0._data[uint(_S6.x + _S6.y * globalParams_0.resolution_0.x)] = raymarch_0(vec3(0.0, 0.0, -3.0), normalize(vec3((vec2(_S6) - 0.5) / float(globalParams_0.resolution_0.x), 1.0)));
    return;
}

