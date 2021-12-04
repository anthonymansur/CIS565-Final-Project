/********************************************************************************************************
 * Adapted from https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss
 * ******************************************************************************************************/

#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices=32) out;
//layout (points, max_vertices=1) out;

uniform mat4 u_projMatrix;

in vec4 geo_v0[];
in vec4 geo_v1[];
in vec2 geo_attrib[];

out float frag_attrib;
out vec3 aNormal;
out float v_coord;

vec3 createPerpendicular(vec3 p1, vec3 p2) {
    vec3 norm = normalize(p2 - p1);
    vec3 ret = cross(norm, vec3(0.0, 0.0, 1.0));

    if (length(ret) == 0.0)
        ret = cross(norm, vec3(0.0, 1.0, 0.0));

    return ret;
}

void main() 
{    
    float r1 = geo_v0[0].w * 2.0;
    float r2 = geo_v1[0].w * 2.0;
    vec4 g0 = vec4(geo_v0[0].x, geo_v0[0].y * 1.35, geo_v0[0].z, geo_v0[0].w);
    vec4 g1 = vec4(geo_v1[0].x, geo_v1[0].y * 1.35, geo_v1[0].z, geo_v1[0].w);
    //geo_v1[0].y *= 2.0;
    //geo_v0[0].y *= 2.0;
    vec3 axis = g1.xyz - g0.xyz;
    vec3 perpX = createPerpendicular(g1.xyz, g0.xyz);
    vec3 perpY = cross(normalize(axis), perpX);

    for (int i = 0; i < 16; i++) {
        float a = i / 15.0 * 2.0 * 3.1415926;
        float ca = cos(a);
        float sa = sin(a);

        vec3 normal = vec3( ca * perpX.x + sa * perpY.x,
                            ca * perpX.y + sa * perpY.y,
                            ca * perpX.z + sa * perpY.z );

        aNormal = normal;

        vec3 p1 = g0.xyz + r1 * normal;
        vec3 p2 = g1.xyz + r2 * normal;

        gl_Position = u_projMatrix * vec4(p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(p2, 1.0);
        EmitVertex();
    }
    EndPrimitive();

   /* if (geo_attrib[0].w > -1.0f) {
        // draw leaf at node 1

    }*/

    /*if (geo_attrib[1].w > -1.0f) {
        // draw leaf at node 2
    }*/

    v_coord = geo_v0[0].z;
}  

