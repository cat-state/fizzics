#version 450

layout(location = 0) out vec2 v_tex_coords;

void main() {
    v_tex_coords = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_tex_coords * 2.0 - 1.0, 0.0, 1.0);
}