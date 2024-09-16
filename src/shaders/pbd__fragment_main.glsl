#version 450
layout(row_major) uniform;
layout(row_major) buffer;

#line 266 0
layout(location = 0)
out vec4 entryPointParam_fragment_main_0;


#line 266
layout(location = 1)
in vec3 mesh_vertex_normal_0;


#line 266
void main()
{

#line 266
    entryPointParam_fragment_main_0 = vec4(max(0.0, dot(normalize(vec3(1.0, 1.0, 1.0)), mesh_vertex_normal_0)) * vec3(1.0, 1.0, 0.0), 1.0);

#line 266
    return;
}

