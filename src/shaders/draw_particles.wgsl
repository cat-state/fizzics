struct Particle {
    x: vec3<f32>,
    mass: f32,
    v: vec3<f32>,
    _padding: f32,
    q: vec4<f32>
}

struct Voxel {
    particles: array<Particle, 8>,
}


struct Uniforms {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

struct MeshVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(mesh_vertex: MeshVertex, @builtin(instance_index) instance_id: u32) -> VertexOutput {
    var output: VertexOutput;
    let particle = particles[instance_id];
    let world_position = mesh_vertex.position + particle.x;
    output.clip_position = uniforms.view_projection * vec4<f32>(world_position, 1.0);
    output.world_position = world_position;
    output.world_normal = mesh_vertex.normal;
    return output;
}

@fragment
fn fragment_main(vertex_output: VertexOutput) -> @location(0) vec4<f32> {
    let light_position = vec3<f32>(10.0, 10.0, 10.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let ambient_strength = 0.1;
    
    let normal = normalize(vertex_output.world_normal);
    let light_dir = normalize(light_position - vertex_output.world_position);
    let view_dir = normalize(uniforms.camera_position.xyz - vertex_output.world_position);
    let half_dir = normalize(light_dir + view_dir);

    let diffuse = max(dot(normal, light_dir), 0.0);
    let specular_strength = 0.5;
    let specular = pow(max(dot(normal, half_dir), 0.0), 32.0) * specular_strength;

    let ambient = ambient_strength * light_color;
    let result = (ambient + diffuse + specular) * vec3<f32>(0.7, 0.7, 1.0);

    return vec4<f32>(result, 1.0);
}