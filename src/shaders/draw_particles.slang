// TODO: use slang modules
struct Particle {
    float3 x;
    float3 v;
};


StructuredBuffer<Particle> particles;
ParameterBlock<float4x4> view_projection;

struct MeshVertex {
    float3 position;
    float3 normal;
};

struct VertexOutput {
    float4 sv_position: SV_Position;
    MeshVertex mesh_vertex;
};

[shader("vertex")]
VertexOutput vertex_main(MeshVertex mesh_vertex, uint instance_id : SV_InstanceID)
{
    VertexOutput output;
    Particle particle = particles[instance_id];
    float3 position = mesh_vertex.position + particle.x;
    output.sv_position = mul(view_projection, float4(position, 1.0));
    output.mesh_vertex = mesh_vertex;
    return output;
}

[shader("fragment")]
float4 fragment_main(MeshVertex mesh_vertex) : SV_Target {
    float3 light_dir = normalize(float3(1, 1, 1));
    float3 normal = mesh_vertex.normal;
    float3 color = max(0, dot(light_dir, normal)) * float3(1, 1, 0);
    return float4(color, 1);
}
