#version 330

in float frag_attrib;
out vec4 outColor;

void main() 
{
    if (frag_attrib > 0.f)
        outColor = vec4(.12, .31, .14, 1.f); // green
    else   
        outColor = vec4(.38, .25, .13, 1.f); // dark brown
}