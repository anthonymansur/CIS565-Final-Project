/********************************************************************************************************
 * Adapted from https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss
 * ******************************************************************************************************/

#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices=100) out;
//layout (points, max_vertices=1) out;
//layout (points, max_vertices=1) out;

uniform mat4 u_projMatrix;

in vec4 geo_v0[];
in vec4 geo_v1[];
in vec3 geo_attrib[];

out vec2 frag_attrib;
out vec3 aNormal;
out float y_coord;

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
    float leaf_r1 = r1 * 5.0;
    float r2 = geo_v1[0].w * 2.0;
    float leaf_r2 = r2 * 5.0;
    vec4 g0 = vec4(geo_v0[0].x, geo_v0[0].y * 1.35, geo_v0[0].z, geo_v0[0].w);
    vec4 g1 = vec4(geo_v1[0].x, geo_v1[0].y * 1.35, geo_v1[0].z, geo_v1[0].w);
    vec3 axis = g1.xyz - g0.xyz;
    vec3 perpX = createPerpendicular(g1.xyz, g0.xyz);
    vec3 perpY = cross(normalize(axis), perpX);
    vec3 norm1;
    vec3 norm2;
    vec3 norm3;
    vec3 norm4;

    for (int i = 0; i < 16; i++) {

        float a = i / 15.0 * 2.0 * 3.1415926;
        float ca = cos(a);
        float sa = sin(a);

        vec3 normal = vec3( ca * perpX.x + sa * perpY.x,
                            ca * perpX.y + sa * perpY.y,
                            ca * perpX.z + sa * perpY.z );

        if (i == 0) {
            norm1 = normal;
        }
        if (i == 4) {
            norm2 = normal;
        }
        if (i == 8) {
            norm3 = normal;
        }
        if (i == 12) {
            norm3 = normal;
        }

        aNormal = normal;

        vec3 p1 = g0.xyz + r1 * normal;
        vec3 p2 = g1.xyz + r2 * normal;

        frag_attrib.x = -1.0f;

        gl_Position = u_projMatrix * vec4(p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(p2, 1.0);
        EmitVertex();
        //frag_attrib.y = geo_attrib[0].z;
    }
    //frag_attrib.y = geo_attrib[0].z; 
    EndPrimitive();
    frag_attrib.y = geo_attrib[0].z; 

   if (geo_attrib[0].x > -1.0f) {
        // draw leaf at node 1
        frag_attrib.x = 1.0f;

        vec3 leaf1_p1 = g0.xyz + r1 * norm1;
        vec3 leaf1_p2 = g0.xyz + (r1 * 3.0) * axis;
        vec3 leaf1_p3 = g0.xyz + (r1 * 5.0) * norm1;
        vec3 leaf1_p4 = g0.xyz - (r1 * -5.0) * axis;

        vec3 leaf2_p1 = g0.xyz + r1 * norm2;
        vec3 leaf2_p2 = g0.xyz + (r1 * 3.0) * axis;
        vec3 leaf2_p3 = g0.xyz + (r1 * 5.0) * norm2;
        vec3 leaf2_p4 = g0.xyz - (r1 * -5.0) * axis;

        vec3 leaf3_p1 = g0.xyz + r1 * norm3;
        vec3 leaf3_p2 = g0.xyz + (r1 * 3.0) * axis;
        vec3 leaf3_p3 = g0.xyz + (r1 * 5.0) * norm3;
        vec3 leaf3_p4 = g0.xyz - (r1 * -5.0) * axis;

        vec3 leaf4_p1 = g0.xyz + r1 * norm4;
        vec3 leaf4_p2 = g0.xyz + (r1 * 3.0) * axis;
        vec3 leaf4_p3 = g0.xyz + (r1 * 5.0) * norm4;
        vec3 leaf4_p4 = g0.xyz - (r1 * -5.0) * axis;

        gl_Position = u_projMatrix * vec4(leaf1_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf1_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf1_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf1_p4, 1.0);
        EmitVertex();

        EndPrimitive();

        gl_Position = u_projMatrix * vec4(leaf2_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf2_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf2_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf2_p4, 1.0);
        EmitVertex();

        EndPrimitive();

        gl_Position = u_projMatrix * vec4(leaf3_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf3_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf3_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf3_p4, 1.0);
        EmitVertex();

        EndPrimitive();

        gl_Position = u_projMatrix * vec4(leaf4_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf4_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf4_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf4_p4, 1.0);
        EmitVertex();

        EndPrimitive();

    }

    if (geo_attrib[0].y > -1.0f) {
        // draw leaf at node 1
        frag_attrib.x = 1.0f;

        vec3 leaf1_p1 = g0.xyz + r2 * norm1;
        vec3 leaf1_p2 = g0.xyz + (r2 * 3.0) * axis;
        vec3 leaf1_p3 = g0.xyz + (r2 * 5.0) * norm1;
        vec3 leaf1_p4 = g0.xyz - (r2 * -5.0) * axis;

        vec3 leaf2_p1 = g0.xyz + r2 * norm2;
        vec3 leaf2_p2 = g0.xyz + (r2 * 3.0) * axis;
        vec3 leaf2_p3 = g0.xyz + (r2 * 5.0) * norm2;
        vec3 leaf2_p4 = g0.xyz - (r2 * -5.0) * axis;

        vec3 leaf3_p1 = g0.xyz + r2 * norm3;
        vec3 leaf3_p2 = g0.xyz + (r2 * 3.0) * axis;
        vec3 leaf3_p3 = g0.xyz + (r2 * 5.0) * norm3;
        vec3 leaf3_p4 = g0.xyz - (r2 * -5.0) * axis;

        vec3 leaf4_p1 = g0.xyz + r2 * norm4;
        vec3 leaf4_p2 = g0.xyz + (r2 * 3.0) * axis;
        vec3 leaf4_p3 = g0.xyz + (r2 * 5.0) * norm4;
        vec3 leaf4_p4 = g0.xyz - (r2 * -5.0) * axis;

        gl_Position = u_projMatrix * vec4(leaf1_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf1_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf1_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf1_p4, 1.0);
        EmitVertex();

        EndPrimitive();

        gl_Position = u_projMatrix * vec4(leaf2_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf2_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf2_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf2_p4, 1.0);
        EmitVertex();

        EndPrimitive();

        gl_Position = u_projMatrix * vec4(leaf3_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf3_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf3_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf3_p4, 1.0);
        EmitVertex();

        EndPrimitive();

        gl_Position = u_projMatrix * vec4(leaf4_p1, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf4_p2, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf4_p3, 1.0);
        EmitVertex();
        gl_Position = u_projMatrix * vec4(leaf4_p4, 1.0);
        EmitVertex();

        EndPrimitive();

    }

    /*if (geo_attrib[0].y > -1.0f) {
        // draw leaf at node 2
    }*/

    y_coord = geo_v0[0].y;
}  
