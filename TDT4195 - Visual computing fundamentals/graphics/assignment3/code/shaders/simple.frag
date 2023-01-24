#version 430 core

in vec4 vertex_color;
in vec3 vertex_normal;

out vec4 frag_color;

// Create a light source
vec3 light_direction = normalize(vec3(0.8, -0.5, 0.6));

void main()
{
    //Task 1c
    //frag_color = vec4(vertex_normal, 1.0f);

    //Task 1d
    frag_color = vec4(vertex_color.xyz * max(0, dot(vertex_normal, -light_direction)), vertex_color.w);
}