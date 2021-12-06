/********************************************************************************************************
 * Adapted from https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss
 * ******************************************************************************************************/

#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices=100) out;

uniform mat4 u_projMatrix;

in vec4 geo_v0[];
in vec4 geo_v1[];
in vec3 geo_attrib[];

out vec2 frag_attrib;
out vec3 frag_normal;
out float frag_height;

// function prototypes
vec3 createPerpendicular(vec3 p1, vec3 p2);

void main() 
{    
    /** Render branch cones */
    // scale radii
    float r1 = geo_v0[0].w * 2.0;
    float r2 = geo_v1[0].w * 2.0;
    // scale height of trees
    vec4 g0 = vec4(geo_v0[0].x, geo_v0[0].y * 1.35, geo_v0[0].z, geo_v0[0].w);
    vec4 g1 = vec4(geo_v1[0].x, geo_v1[0].y * 1.35, geo_v1[0].z, geo_v1[0].w);
    // find the axis and tangent vectors
    vec3 axis = g1.xyz - g0.xyz;
    vec3 perpX = createPerpendicular(g1.xyz, g0.xyz);
    vec3 perpY = cross(normalize(axis), perpX);

    // subdivision
    for (int i = 0; i < 16; i++) {

        float a = i / 15.0 * 2.0 * 3.1415926;
        float ca = cos(a);
        float sa = sin(a);

        vec3 normal = vec3( ca * perpX.x + sa * perpY.x,
                            ca * perpX.y + sa * perpY.y,
                            ca * perpX.z + sa * perpY.z );

        frag_normal = normal;

        vec3 p1 = g0.xyz + r1 * normal;
        vec3 p2 = g1.xyz + r2 * normal;

        // Generate vertices
        gl_Position = u_projMatrix * vec4(p1, 1.0);
        frag_attrib.x = -1.0f;
        frag_attrib.y = geo_attrib[0].z;
        frag_height = geo_v0[0].y;
        EmitVertex();

        gl_Position = u_projMatrix * vec4(p2, 1.0);
        frag_attrib.x = -1.0f;
        frag_attrib.y = geo_attrib[0].z;
        frag_height = geo_v1[0].y;
        EmitVertex();
    }

    /** Render Leaves */
    // TODO: implement
}  

// function implementations
vec3 createPerpendicular(vec3 p1, vec3 p2) {
    vec3 norm = normalize(p2 - p1);
    vec3 ret = cross(norm, vec3(0.0, 0.0, 1.0));

    if (length(ret) == 0.0)
        ret = cross(norm, vec3(0.0, 1.0, 0.0));

    return ret;
}
