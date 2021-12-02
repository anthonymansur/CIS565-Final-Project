#version 330 core
layout (points) in;
//layout (triangle_strip, max_vertices=64) out;
layout (points, max_vertices=1) out;

in vec4 in_v0[];
in vec4 in_v1[];

uniform mat4 u_projMatrix;

void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    gl_Position = u_projMatrix * vec4(Position, 1.0);
    EmitVertex();
    EndPrimitive();
}

/*
void main() 
{    
    vec3 axis = normalize(in_v1[0].xyz - in_v0[0].xyz);
    vec3 orth;
    if (abs(axis.x) > abs(axis.z))
        orth = vec3(-axis.y, axis.x, 0.f);
    else
        orth = vec3(0.f, -axis.z, axis.y);
    orth = normalize(orth);

    // first triangle
    gl_Position = u_projMatrix * vec4((in_v0[0].xyz + (orth * in_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((in_v0[0].xyz - (orth * in_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((in_v1[0].xyz - (orth * in_v0[0].w)), 1.f);
    EmitVertex();
    EndPrimitive();

    // second triangle
    gl_Position = u_projMatrix * vec4((in_v0[0].xyz + (orth * in_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((in_v1[0].xyz + (orth * in_v0[0].w)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((in_v1[0].xyz - (orth * in_v0[0].w)), 1.f);
    EmitVertex();
    EndPrimitive();
}  
*/
