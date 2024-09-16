#version 450
layout(row_major) uniform;
layout(row_major) buffer;

#line 4 0
struct Particle_std430_0
{
    vec3 x_0;
    vec3 v_0;
};


#line 8
layout(std430, binding = 0) readonly buffer StructuredBuffer_Particle_std430_t_0 {
    Particle_std430_0 _data[];
} particles_0;

#line 1313 1
struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    vec4  data_0[4];
};


#line 9 0
layout(binding = 0, set = 1)
layout(std140) uniform _S1
{
    vec4  data_0[4];
}view_projection_0;

#line 9
mat4x4 unpackStorage_0(_MatrixStorage_float4x4_ColMajorstd140_0 _S2)
{

#line 9
    return mat4x4(_S2.data_0[0][0], _S2.data_0[1][0], _S2.data_0[2][0], _S2.data_0[3][0], _S2.data_0[0][1], _S2.data_0[1][1], _S2.data_0[2][1], _S2.data_0[3][1], _S2.data_0[0][2], _S2.data_0[1][2], _S2.data_0[2][2], _S2.data_0[3][2], _S2.data_0[0][3], _S2.data_0[1][3], _S2.data_0[2][3], _S2.data_0[3][3]);
}


#line 2
struct Particle_0
{
    vec3 x_0;
    vec3 v_0;
};


#line 2
Particle_0 unpackStorage_1(Particle_std430_0 _S3)
{

#line 2
    Particle_0 _S4 = { _S3.x_0, _S3.v_0 };

#line 2
    return _S4;
}


#line 11183 2
layout(location = 0)
out vec3 entryPointParam_vertex_main_mesh_vertex_position_0;


#line 11183
layout(location = 1)
out vec3 entryPointParam_vertex_main_mesh_vertex_normal_0;


#line 11183
layout(location = 0)
in vec3 mesh_vertex_position_0;


#line 11183
layout(location = 1)
in vec3 mesh_vertex_normal_0;


#line 11 0
struct MeshVertex_0
{
    vec3 position_0;
    vec3 normal_0;
};


#line 16
struct VertexOutput_0
{
    vec4 sv_position_0;
    MeshVertex_0 mesh_vertex_0;
};

void main()
{
    VertexOutput_0 output_0;


    output_0.sv_position_0 = (((vec4(mesh_vertex_position_0 + unpackStorage_1(particles_0._data[uint(uint(gl_InstanceIndex))]).x_0, 1.0)) * (unpackStorage_0(view_projection_0))));

#line 27
    output_0.mesh_vertex_0.position_0 = mesh_vertex_position_0;

#line 27
    output_0.mesh_vertex_0.normal_0 = mesh_vertex_normal_0;

    VertexOutput_0 _S5 = output_0;

#line 29
    gl_Position = output_0.sv_position_0;

#line 29
    entryPointParam_vertex_main_mesh_vertex_position_0 = _S5.mesh_vertex_0.position_0;

#line 29
    entryPointParam_vertex_main_mesh_vertex_normal_0 = _S5.mesh_vertex_0.normal_0;

#line 29
    return;
}

