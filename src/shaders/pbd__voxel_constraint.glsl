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

#line 202
struct GlobalParams_std140_0
{
    vec4 i_mouse_0;
    vec2 i_resolution_0;
    int i_frame_0;
    uvec3 dim_0;
};


#line 202
layout(binding = 0)
layout(std140) uniform _S1
{
    vec4 i_mouse_0;
    vec2 i_resolution_0;
    int i_frame_0;
    uvec3 dim_0;
}globalParams_0;

#line 13
struct Particle_0
{
    vec3 x_0;
    vec3 v_0;
};


#line 13
Particle_0 unpackStorage_0(Particle_std430_0 _S2)
{

#line 13
    Particle_0 _S3 = { _S2.x_0, _S2.v_0 };

#line 13
    return _S3;
}



struct Voxel_0
{
    Particle_0  particles_1[8];
};




Voxel_0 get_voxel_0(uint idx_0)
{
    uint _S4 = idx_0 * 8U;

#line 27
    Particle_0  _S5[8] = { unpackStorage_0(particles_0._data[uint(_S4)]), unpackStorage_0(particles_0._data[uint(_S4 + 1U)]), unpackStorage_0(particles_0._data[uint(_S4 + 2U)]), unpackStorage_0(particles_0._data[uint(_S4 + 3U)]), unpackStorage_0(particles_0._data[uint(_S4 + 4U)]), unpackStorage_0(particles_0._data[uint(_S4 + 5U)]), unpackStorage_0(particles_0._data[uint(_S4 + 6U)]), unpackStorage_0(particles_0._data[uint(_S4 + 7U)]) };

#line 27
    Voxel_0 _S6 = { _S5 };

#line 27
    return _S6;
}


#line 119
void handle_boundary_collisions_0(inout vec3 position_0, inout vec3 velocity_0)
{

#line 120
    if(position_0.x < -1.0)
    {

#line 121
        position_0[0] = -1.0;
        velocity_0[0] = velocity_0[0] * -0.00999999977648258;

#line 120
    }
    else
    {
        if(position_0.x > 1.0)
        {

#line 124
            position_0[0] = 1.0;
            velocity_0[0] = velocity_0[0] * -0.00999999977648258;

#line 123
        }

#line 120
    }

#line 128
    if(position_0.y < -0.5)
    {

#line 129
        position_0[1] = -0.5;
        velocity_0[1] = velocity_0[1] * -0.00999999977648258;

#line 128
    }
    else
    {
        if(position_0.y > 2.0)
        {

#line 132
            position_0[1] = 0.5;
            velocity_0[1] = velocity_0[1] * -0.00999999977648258;

#line 131
        }

#line 128
    }

#line 136
    if(position_0.z < -1.0)
    {

#line 137
        position_0[2] = -1.0;
        velocity_0[2] = velocity_0[2] * -0.00999999977648258;

#line 136
    }
    else
    {
        if(position_0.z > 1.0)
        {

#line 140
            position_0[2] = 1.0;
            velocity_0[2] = velocity_0[2] * -0.00999999977648258;

#line 139
        }

#line 136
    }

#line 143
    return;
}


#line 45
vec3 project_0(vec3 a_0, vec3 b_0)
{

#line 46
    return dot(a_0, b_0) / dot(b_0, b_0) * b_0;
}

void gram_schmidt_0(vec3 A_0, vec3 B_0, vec3 C_0, out vec3 Ao_0, out vec3 Bo_0, out vec3 Co_0)
{

#line 50
    vec3 _S7 = normalize(A_0);

#line 50
    Ao_0 = _S7;

    vec3 _S8 = normalize(B_0 - project_0(B_0, _S7));

#line 52
    Bo_0 = _S8;

    Co_0 = normalize(C_0 - project_0(C_0, Ao_0) - project_0(C_0, _S8));
    return;
}


#line 57
vec3 slerp_0(vec3 start_0, vec3 end_0, float t_0)
{

#line 58
    float cos_theta_0 = dot(start_0, end_0);

    if(cos_theta_0 > 0.99949997663497925)
    {

#line 61
        return normalize(mix(start_0, end_0, vec3(t_0)));
    }


    float theta_0 = acos(clamp(cos_theta_0, -1.0, 1.0));
    float sin_theta_0 = sin(theta_0);

#line 71
    return normalize(sin((1.0 - t_0) * theta_0) / sin_theta_0 * start_0 + sin(t_0 * theta_0) / sin_theta_0 * end_0);
}

vec3 average_on_sphere_0(vec3 v1_0, vec3 v2_0, vec3 v3_0)
{
    return slerp_0(slerp_0(v1_0, v2_0, 0.5), v3_0, 0.3333333432674408);
}

void apply_gram_schmidt_constraint_0(inout Voxel_0 voxel_0)
{

#line 79
    int i_0 = 0;

#line 79
    for(;;)
    {

#line 80
        if(i_0 < 8)
        {
        }
        else
        {

#line 80
            break;
        }



        vec3 edge1_0 = voxel_0.particles_1[(i_0 + 1) % 4 + i_0 / 4 * 4].x_0 - voxel_0.particles_1[i_0].x_0;
        vec3 edge2_0 = voxel_0.particles_1[(i_0 + 3) % 4 + i_0 / 4 * 4].x_0 - voxel_0.particles_1[i_0].x_0;
        vec3 edge3_0 = voxel_0.particles_1[i_0 ^ 4].x_0 - voxel_0.particles_1[i_0].x_0;

        vec3 u1_0;

#line 89
        vec3 u2_0;

#line 89
        vec3 u3_0;

        gram_schmidt_0(edge1_0, edge2_0, edge3_0, u1_0, u2_0, u3_0);

#line 89
        vec3 v1_1;

#line 89
        vec3 v2_1;

#line 89
        vec3 v3_1;


        gram_schmidt_0(edge2_0, edge3_0, edge1_0, v1_1, v2_1, v3_1);

#line 89
        vec3 w1_0;

#line 89
        vec3 w2_0;

#line 89
        vec3 w3_0;



        gram_schmidt_0(edge3_0, edge1_0, edge2_0, w1_0, w2_0, w3_0);

        vec3 _S9 = average_on_sphere_0(u1_0, w2_0, v3_1);
        vec3 _S10 = average_on_sphere_0(u2_0, v1_1, w3_0);
        vec3 _S11 = average_on_sphere_0(u3_0, v2_1, w1_0);

        u1_0 = _S9;
        u2_0 = _S10;
        u3_0 = _S11;

#line 107
        vec3 correction_next1_0 = (voxel_0.particles_1[i_0].x_0 + _S9 - voxel_0.particles_1[(i_0 + 1) % 4 + i_0 / 4 * 4].x_0) * 0.5;
        vec3 correction_next2_0 = (voxel_0.particles_1[i_0].x_0 + _S10 - voxel_0.particles_1[(i_0 + 3) % 4 + i_0 / 4 * 4].x_0) * 0.5;
        vec3 correction_next3_0 = (voxel_0.particles_1[i_0].x_0 + _S11 - voxel_0.particles_1[i_0 ^ 4].x_0) * 0.5;
        vec3 correction_self_0 = - (correction_next1_0 + correction_next2_0 + correction_next3_0) / 3.0;

        voxel_0.particles_1[(i_0 + 1) % 4 + i_0 / 4 * 4].x_0 = voxel_0.particles_1[(i_0 + 1) % 4 + i_0 / 4 * 4].x_0 + correction_next1_0;
        voxel_0.particles_1[(i_0 + 3) % 4 + i_0 / 4 * 4].x_0 = voxel_0.particles_1[(i_0 + 3) % 4 + i_0 / 4 * 4].x_0 + correction_next2_0;
        voxel_0.particles_1[i_0 ^ 4].x_0 = voxel_0.particles_1[i_0 ^ 4].x_0 + correction_next3_0;
        voxel_0.particles_1[i_0].x_0 = voxel_0.particles_1[i_0].x_0 + correction_self_0;

#line 80
        i_0 = i_0 + 1;

#line 80
    }

#line 117
    return;
}


#line 202
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 203
    uint cube_index_0 = gl_GlobalInvocationID.x;

    Voxel_0 voxel_1 = get_voxel_0(cube_index_0);
    vec3 mouse_pos_0 = vec3((globalParams_0.i_mouse_0.xy - 0.5 * globalParams_0.i_resolution_0.xy) / globalParams_0.i_resolution_0.y, 0.0);

#line 206
    int i_1 = 0;

#line 206
    for(;;)
    {

#line 214
        if(i_1 < 8)
        {
        }
        else
        {

#line 214
            break;
        }
        vec3 velocity_1 = voxel_1.particles_1[i_1].v_0;
        vec3 new_pos_0 = voxel_1.particles_1[i_1].x_0 + voxel_1.particles_1[i_1].v_0 * 0.01666666753590107 + vec3(0.0, -9.80000019073486328, 0.0) * 0.01666666753590107 * 0.01666666753590107;


        handle_boundary_collisions_0(new_pos_0, velocity_1);
        voxel_1.particles_1[i_1].x_0 = new_pos_0;
        voxel_1.particles_1[i_1].v_0 = velocity_1;
        if(i_1 == -1)
        {
            new_pos_0 = new_pos_0 + (mouse_pos_0 - new_pos_0) * 10.0 * 0.01666666753590107;

#line 223
        }

#line 214
        i_1 = i_1 + 1;

#line 214
    }

#line 230
    apply_gram_schmidt_constraint_0(voxel_1);

#line 230
    i_1 = 0;

#line 230
    for(;;)
    {
        if(i_1 < 8)
        {
        }
        else
        {

#line 232
            break;
        }

#line 233
        int particle_index_0 = int(cube_index_0) * 8 + i_1;
        vec3 new_pos_1 = voxel_1.particles_1[i_1].x_0;
        vec3 velocity_2 = voxel_1.particles_1[i_1].v_0;
        handle_boundary_collisions_0(new_pos_1, velocity_2);

        particles_0._data[uint(particle_index_0)].x_0 = new_pos_1;
        particles_0._data[uint(particle_index_0)].v_0 = (new_pos_1 - particles_0._data[uint(particle_index_0)].x_0) / 0.01666666753590107;

#line 232
        i_1 = i_1 + 1;

#line 232
    }

#line 241
    return;
}

