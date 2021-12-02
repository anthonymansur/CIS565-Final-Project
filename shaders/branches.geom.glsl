#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices=6) out;
//layout (points, max_vertices=1) out;

uniform mat4 u_projMatrix;

in vec4 geo_v0[];
in vec4 geo_v1[];
in vec2 geo_attrib[];

out float frag_attrib;
/*
void main()
{
    frag_attrib = geo_attrib[0][1];
    vec3 pos = gl_in[0].gl_Position.xyz;
    gl_Position = u_projMatrix * vec4(pos, 1.0);
    EmitVertex();
    EndPrimitive();
}
*/

void main() 
{    
    vec3 axis = normalize(geo_v1[0].xyz - geo_v0[0].xyz);
    vec3 orth;
    if (abs(axis.x) > abs(axis.z))
        orth = vec3(-axis.y, axis.x, 0.f);
    else
        orth = vec3(0.f, -axis.z, axis.y);
    orth = normalize(orth);

    // first triangle
    gl_Position = u_projMatrix * vec4((geo_v0[0].xyz + (orth * geo_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v0[0].xyz - (orth * geo_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v1[0].xyz - (orth * geo_v1[0].w)), 1.f);
    EmitVertex();
    EndPrimitive();

    // second triangle
    gl_Position = u_projMatrix * vec4((geo_v0[0].xyz + (orth * geo_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v1[0].xyz + (orth * geo_v1[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v1[0].xyz - (orth * geo_v1[0].w)), 1.f);
    EmitVertex();
    EndPrimitive();
}  

