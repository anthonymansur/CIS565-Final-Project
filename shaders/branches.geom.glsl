/********************************************************************************************************
 * Adapted from https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss
 * ******************************************************************************************************/

#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices=102) out;

uniform mat4 u_projMatrix;

in vec4 geo_v0[];
in vec4 geo_v1[];
in vec3 geo_attrib[];

out vec2 frag_attrib;
out vec3 frag_normal;
out float frag_height;

// function prototypes
vec3 createPerpendicular(vec3 p1, vec3 p2);
vec3 genNormal(vec3 p, vec3 axis, int seed);
float noise1D(vec2 p);
float noise1D(float a, float b);

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
    vec3 axis = normalize(g1.xyz - g0.xyz);
    vec3 perpX = createPerpendicular(g1.xyz, g0.xyz);
    vec3 perpY = cross(axis, perpX);

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
    EndPrimitive();
    
    if (geo_attrib[0].x > 0 && geo_attrib[0].y > 0)
    {
        /** Render Leaves */
        float rand1 = noise1D(g1.x * g1.y * g1.z, g0.x * g0.y * g0.z);
        float rand2 = noise1D(g0.x * g0.y * g0.z, g1.x * g1.y * g1.z);

        int leavesPerUnitLength = 200; // TODO: add some noise
        float leaf_length = 0.05 + 0.1 * rand1; // TODO: add some noise
        int leaf_color;

        if (rand2 < 0.55)
            leaf_color = 0;
        else if (rand2 < 0.85)
            leaf_color = 1;
        else 
            leaf_color = 2;

        float len = distance(g0, g1);
        for (int i = 0; i < (leavesPerUnitLength * distance(g0.xyz, g1.xyz)); i++)
        {
            // calculate the leaf axis
            float dist = i / (leavesPerUnitLength * distance(g0.xyz, g1.xyz)); // TODO: add noise here
            vec3 posAlongBranch = g0.xyz + len * dist;
            vec3 leaf_axis = genNormal(posAlongBranch, axis, i);

            // calculate the leaf start position
            vec3 orth = cross(axis, leaf_axis);
            vec3 startPos = posAlongBranch /*+ orth * (r1 + (r2 - r1) * dist)*/;
            float offset = 0.5 * noise1D(g0.x * g0.y * g0.z * i, g1.x * g1.y * g1.z * i);
            startPos = startPos + orth * offset; // offset;

            // calculate the leaf normal
            vec3 leaf_norm = genNormal(startPos, leaf_axis, i);
            vec3 leaf_orth = cross(leaf_axis, leaf_norm);
            
            // calculate the vertices that make up the leaf
            // TODO: create different shapes
            // diamond shape
            vec3 pos1 = startPos + leaf_axis * (leaf_length / 2) - leaf_orth *  (leaf_length / 2);
            vec3 pos2 = startPos + leaf_axis * (leaf_length / 2) + leaf_orth *  (leaf_length / 2);
            vec3 pos3 = startPos + leaf_axis * leaf_length;
            gl_Position = u_projMatrix * vec4(startPos, 1.0);
            frag_attrib.x = 1.0f;
            frag_attrib.y = leaf_color;
            EmitVertex();
            gl_Position = u_projMatrix * vec4(pos1, 1.0);
            frag_attrib.x = 1.0f;  
            frag_attrib.y = leaf_color;     
            EmitVertex();
            gl_Position = u_projMatrix * vec4(pos3, 1.0);
            frag_attrib.x = 1.0f;    
            frag_attrib.y = leaf_color;   
            EmitVertex();

            gl_Position = u_projMatrix * vec4(startPos, 1.0);
            frag_attrib.x = 1.0f;
            frag_attrib.y = leaf_color;
            EmitVertex();
            gl_Position = u_projMatrix * vec4(pos2, 1.0);
            frag_attrib.x = 1.0f;    
            frag_attrib.y = leaf_color;   
            EmitVertex();
            gl_Position = u_projMatrix * vec4(pos3, 1.0);
            frag_attrib.x = 1.0f;    
            frag_attrib.y = leaf_color;   
            EmitVertex();

            EndPrimitive();
        }
    }
}  

// function implementations
vec3 createPerpendicular(vec3 p1, vec3 p2) {
    vec3 norm = normalize(p2 - p1);
    vec3 ret = cross(norm, vec3(0.0, 0.0, 1.0));

    if (length(ret) == 0.0)
        ret = cross(norm, vec3(0.0, 1.0, 0.0));

    return ret;
}

// Taken from CIS 460
float noise1D(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise1D(float a, float b)
{
    return noise1D(vec2(a, b));
}

vec3 genNormal(vec3 p, vec3 axis, int seed)
{
    vec3 tangent = vec3(noise1D(p.x, seed), noise1D(p.y, seed), noise1D(p.z, seed));
    return cross(axis, normalize(tangent));
}