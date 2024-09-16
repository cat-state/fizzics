#version 450
layout(row_major) uniform;
layout(row_major) buffer;

#line 15 0
struct Particle_std430_0
{
    vec3 x_0;
    vec3 v_0;
};


#line 39
layout(std430, binding = 1) buffer StructuredBuffer_Particle_std430_t_0 {
    Particle_std430_0 _data[];
} particles_0;

#line 13
struct Particle_0
{
    vec3 x_0;
    vec3 v_0;
};


#line 13
Particle_0 unpackStorage_0(Particle_std430_0 _S1)
{

#line 13
    Particle_0 _S2 = { _S1.x_0, _S1.v_0 };

#line 13
    return _S2;
}


#line 13
Particle_std430_0 packStorage_0(Particle_0 _S3)
{

#line 13
    Particle_std430_0 _S4 = { _S3.x_0, _S3.v_0 };

#line 13
    return _S4;
}


#line 148
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 149
    uint particle_index_0 = gl_GlobalInvocationID.x;
    Particle_0 p_i_0 = unpackStorage_0(particles_0._data[uint(particle_index_0)]);

#line 150
    int j_0 = int(particle_index_0 + 1U);

#line 150
    for(;;)
    {

#line 151
        if(j_0 < 24)
        {
        }
        else
        {

#line 151
            break;
        }

#line 152
        if(int(particle_index_0) / 8 == j_0 / 8)
        {

#line 152
            j_0 = j_0 + 1;

#line 152
            continue;
        }
        Particle_0 p_j_0 = unpackStorage_0(particles_0._data[uint(j_0)]);
        vec3 diff_0 = p_i_0.x_0 - p_j_0.x_0;
        float distance_0 = length(diff_0);

        if(distance_0 < 0.37999999523162842)
        {

#line 159
            vec3 collision_normal_0 = normalize(diff_0);
            float overlap_0 = 0.37999999523162842 - distance_0;

#line 166
            float velocity_along_normal_0 = dot(p_i_0.v_0 - p_j_0.v_0, collision_normal_0);

            if(velocity_along_normal_0 > 0.0)
            {

#line 168
                j_0 = j_0 + 1;

#line 168
                continue;
            }

#line 176
            vec3 _S5 = -1.5 * velocity_along_normal_0 / 2.0 * collision_normal_0 + overlap_0 * 0.5 * collision_normal_0;

#line 176
            p_i_0.x_0 = p_i_0.x_0 + _S5;
            p_j_0.x_0 = p_j_0.x_0 - _S5;

            particles_0._data[uint(j_0)] = packStorage_0(p_j_0);

#line 158
        }

#line 151
        j_0 = j_0 + 1;

#line 151
    }

#line 183
    particles_0._data[uint(particle_index_0)] = packStorage_0(p_i_0);
    return;
}

