#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) buffer OutputBuffer {
    vec4 data[];
} output_buffer;

layout(set = 0, binding = 1) uniform Dimensions {
    uvec2 dimensions;
};

void main() {
    uvec2 pixel_coords = uvec2(v_tex_coords * vec2(dimensions));
    uint index = pixel_coords.y * dimensions.x + pixel_coords.x;
    f_color = output_buffer.data[index];
}