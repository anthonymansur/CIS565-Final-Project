#version 330

in float frag_attrib;
in float v_coord;
in vec3 aNormal;
out vec4 outColor;

void main() 
{
    // if (frag_attrib > 0.f)
    //     outColor = vec4(.12, .31, .14, 1.f); // green
    // else
    vec4 color = vec4(.38, .25, .13, 1.f); // dark brown
    //float multiplier = (0.5 - v_coord) * 0.05 + v_coord * 1.0;
    outColor = /*multiplier **/ color;

}