struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    let tex_coords_raw = vec2<f32>(u, v);
    
    let position = vec4<f32>(tex_coords_raw * 2.0 - 1.0, 0.0, 1.0);
    let tex_coords = vec2<f32>(tex_coords_raw.x, 1.0 - tex_coords_raw.y); // Invert y-axis due to WGPU's y-axis being up not down.    
    var out = VertexOutput();
    out.clip_position = vec4<f32>(position.x, position.y, 0.0, 1.0);
    out.tex_coords = tex_coords;
    return out;
}