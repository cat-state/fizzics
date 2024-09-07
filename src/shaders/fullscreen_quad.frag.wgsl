struct Dimensions {
    dimensions: vec2<u32>,
}

@group(0) @binding(1) var<storage, read_write> output_buffer: array<vec4<f32>>;
@group(0) @binding(0) var<uniform> dimensions: Dimensions;

@fragment
fn main(@location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let pixel_coords = vec2<u32>(tex_coords * vec2<f32>(dimensions.dimensions));
    let index = pixel_coords.y * dimensions.dimensions.x + pixel_coords.x;
    return output_buffer[index];
}