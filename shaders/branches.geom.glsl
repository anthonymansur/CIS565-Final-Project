#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices=32) out;
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

vec3 createPerpendicular(vec3 p1, vec3 p2) {
    vec3 norm = normalize(p2 - p1);
    vec3 ret = cross(norm, vec3(0.0, 0.0, 1.0));

    if (length(ret) == 0.0)
        ret = cross(norm, vec3(0.0, 1.0, 0.0));

    return ret;
}

void main() 
{    
    //vec3 axis = normalize(geo_v1[0].xyz - geo_v0[0].xyz);
    /*vec3 orth;
    if (abs(axis.x) > abs(axis.z))
        orth = vec3(-axis.y, axis.x, 0.f);
    else
        orth = vec3(0.f, -axis.z, axis.y);
    orth = normalize(orth);*/

    //vec3 radius1 = orth * geo_v0[0].w;
    //vec3 radius2 = orth * geo_v1[0].w;

    float r1 = geo_v0[0].w;
    float r2 = geo_v1[0].w;
    vec3 axis = geo_v1[0].xyz - geo_v0[0].xyz;
    vec3 perpX = createPerpendicular(geo_v1[0].xyz, geo_v0[0].xyz);
    vec3 perpY = cross(normalize(axis), perpX);

    for (int i = 0; i < 16; i++) {
        float a = i / 15.0 * 2.0 * 3.1415926;
        float ca = cos(a);
        float sa = sin(a);

        vec3 normal = vec3( ca * perpX.x + sa * perpY.x,
                            ca * perpX.y + sa * perpY.y,
                            ca * perpX.z + sa * perpY.z );

        vec3 p1 = geo_v0[0].xyz + r1 * normal;
        vec3 p2 = geo_v1[0].xyz + r2 * normal;

        gl_Position = u_projMatrix * vec4(p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(p2, 1.0);
        EmitVertex();
    }
    EndPrimitive();

    // first triangle
    /*gl_Position = u_projMatrix * vec4((geo_v0[0].xyz + (radius1)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v0[0].xyz - (radius1)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v1[0].xyz - (radius2)), 1.f);
    EmitVertex();
    EndPrimitive();*/

    // second triangle
    /*gl_Position = u_projMatrix * vec4((geo_v0[0].xyz + (radius1)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v1[0].xyz + (radius2)), 1.f);
    EmitVertex();
    gl_Position = u_projMatrix * vec4((geo_v1[0].xyz - (radius2)), 1.f);
    EmitVertex();
    EndPrimitive();*/
}  
